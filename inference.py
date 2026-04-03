import collections, math, json, copy, re, os, time
from dataclasses import asdict, dataclass, field
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import transformers
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoProcessor
from torchvision.io import read_video

from qwen_vl_utils import process_vision_info
from model import Qwen2_5_VLForConditionalGeneration
import logging

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import speculative_generate_for_streaming, ar_generate_for_streaming

logger = transformers.logging.get_logger('inference')
logger.setLevel(logging.INFO)


@dataclass
class ProactiveTestArguments(TrainingArguments):
    llm_pretrained: str = 'Qwen/Qwen2.5-VL-3B-Instruct'
    attn_implementation: str = 'flash_attention_2'
    system_prompt: str = "You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames. Your responses should include information from the video since your last reply (if any). If the information in this segment of the video cannot answer the question, output \"NO REPLY\"."
    is_online_model: bool = True
    input_assistant_turns: bool = False
    test_fname: str = ''
    output_fname: str = ''
    start_idx: int = 0
    end_idx: int = None

    # generation arguments
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 40

    # speculative decoding arguments
    use_speculative_decoding: bool = False
    compare_with_baseline: bool = False  # Run both AR and speculative, compare outputs & speed
    adapter_path: str = None
    exit_layer: int = 2
    speculative_threshold: float = 0.6
    speculative_steps: int = 6
    num_adapter_layers: int = 1
    verify_mode: str = 'strict'  # 'strict' = lossless sequential verify, 'fast' = block verify


def get_args():
    args, = HfArgumentParser(ProactiveTestArguments).parse_args_into_dataclasses()
    return args


# tailored for timechat-online (or, say, Qwen-2.5 VL)
class ProactiveInferenceClient:
    def __init__(self, args=None, model=None, processor=None) -> None:
        self.args = args
        self.use_speculative_decoding = getattr(args, 'use_speculative_decoding', False)
        self.compare_with_baseline = getattr(args, 'compare_with_baseline', False)

        if self.use_speculative_decoding and model is None:
            logger.info("Loading model with speculative decoding (Kangaroo adapter)")
            self.kangaroo_model = KangarooQwenModel(
                base_model_path=args.llm_pretrained,
                adapter_model_path=args.adapter_path,
                early_exit_layer=args.exit_layer,
                dtype=torch.bfloat16,
                attn_implementation=args.attn_implementation,
                num_adapter_layers=args.num_adapter_layers,
            ).to('cuda:0')
            self.model = self.kangaroo_model.base_model.model  # raw Qwen2.5-VL model for compatibility
            self.speculative_threshold = args.speculative_threshold
            self.speculative_steps = args.speculative_steps
            self.exit_layer = args.exit_layer
            self.verify_mode = getattr(args, 'verify_mode', 'strict')
        else:
            self.kangaroo_model = None
            self.model = model if model is not None else Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.llm_pretrained, torch_dtype=torch.bfloat16, attn_implementation=args.attn_implementation,
            ).eval().to('cuda:0')

        self.processor = processor if processor is not None else AutoProcessor.from_pretrained(
            args.llm_pretrained
        )
        self.system_prompt = args.system_prompt
        logger.info("using system prompt:" + self.system_prompt)
        self.input_assistant_turns = args.input_assistant_turns
        logger.info(f"using assistant turns in input: {self.input_assistant_turns}")

        self.do_sample = args.do_sample
        self.temperature = args.temperature
        self.top_k = args.top_k

        self.history = list()
        self.prev_frame_before_token_drop = None    # for dynamic token drop
        self.must_reply_prompt = "I must reply.\n"
        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        # Generation speed tracking
        self.generation_stats = []
        self.reset()

    def set_fps(self, fps=None, frame_interval=None):
        assert fps is not None or frame_interval is not None
        assert not (fps is not None and frame_interval is not None)
        if fps is not None:
            self.frame_fps = fps
            self.frame_interval = 1 / self.frame_fps
        else:
            self.frame_interval = frame_interval
            self.frame_fps = 1 / self.frame_interval

    def reset(self, ):
        self.query_queue = collections.deque()
        self.frame_embeds_queue = collections.deque()
        self.video_time = 0
        self.frame_idx = 0
        self.video_tensor = None
        self.past_key_values = None
        self.past_key_values_ar = None  # Separate KV cache for AR baseline comparison
        self.history = list()
        self.prev_frame_before_token_drop = None

        self.prev_image_inputs = list()
        self.prev_video_inputs = list()
        self.all_keep_masks = list()
        self.generation_stats = []
        if hasattr(self.model, 'reset_status'):
            self.model.reset_status()
        if self.kangaroo_model is not None:
            self.kangaroo_model.reset_status()

    def input_query_stream(self, conversation):
        if conversation[0]['role'] != 'system':
            self.query_queue.append({'role': 'system', 'content': self.system_prompt})
        else:
            logger.info(f"using system prompt in data instead of default system prompt: {conversation[0]['content']=}")
            self.query_queue.append(conversation[0])
            del conversation[0]
        for turn in conversation:
            if self.input_assistant_turns or turn['role'] == 'user':
                self.query_queue.append(turn)

    def _recursive_stat_num_frames(self, inputs):
        num_frames = 0
        if isinstance(inputs, (list, tuple)):
            for input in inputs:
                if isinstance(input, (torch.Tensor, Image.Image, np.ndarray)):
                    num_frames += 1
                elif isinstance(input, (list, tuple)):
                    num_frames += self._recursive_stat_num_frames(input)
        return num_frames
            
    def _encode_query(self, debug_print=False):
        newly_added_turns = list()
        while True:
            query = self.query_queue.popleft()
            self.history.append(query)
            newly_added_turns.append(query)
            # we don't need reply after system prompt, or the test data clearly specified that we do not need to reply now
            if query['role'] in ['system', 'assistant'] or query.get('skip_inference', False):
                pass
            # otherwise, we need to break the data loading loop, let the model encode the conversation turns loaded and start to generate replies
            else:
                break

        text = self.processor.apply_chat_template(
            self.history, tokenize=False, add_generation_prompt=True,
        )

        # Check if the last turn in the newly added turns explicitly requires a reply
        if query.get('must_reply', False):
            text += self.must_reply_prompt

        if debug_print:
            print("DEBUG text before generate:", text)

        # Theoretically, the most standard approach should be:
        # image_inputs, video_inputs = process_vision_info(self.history)
        # This ensures that the loaded image_inputs, video_inputs correspond to the image placeholders in the text.
        # However, the images and videos from previous turns have already been loaded before, so in this round we only need to load the latest turn's images and videos and merge them with the previous ones.
        # In fact, the images and videos from previous rounds won't be used in this round's generate... because they have already been encoded by the LLM and placed in the KVCache. But to avoid errors, we still need to ensure that the images and videos from previous rounds are passed in, at least formally.
        new_image_inputs, new_video_inputs = process_vision_info(newly_added_turns)
        if new_image_inputs is not None:
            self.prev_image_inputs.extend(new_image_inputs)
        if new_video_inputs is not None:
            self.prev_video_inputs.extend(new_video_inputs)
        image_inputs = copy.deepcopy(self.prev_image_inputs) if self.prev_image_inputs else None
        video_inputs = copy.deepcopy(self.prev_video_inputs) if self.prev_video_inputs else None

        num_frames = self._recursive_stat_num_frames(new_image_inputs) + self._recursive_stat_num_frames(new_video_inputs)
        self.video_time += num_frames * self.frame_interval
        self.history[-1]['time'] = self.video_time

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda:0")

        if self.model.model.all_keep_masks:
            assert inputs.input_ids.size(0) == 1, "token drop in inference only support batch size 1 now"
            # Drop the visual tokens that were already dropped in previous rounds in the current input_ids as well
            keep_mask = torch.ones_like(inputs.input_ids, dtype=torch.bool)
            old_keep_mask = torch.cat(self.model.model.all_keep_masks, dim=1)
            keep_mask[:, :old_keep_mask.size(1)] = old_keep_mask
            inputs['input_ids'] = inputs.input_ids[keep_mask].unsqueeze(0)
            inputs['attention_mask'] = inputs.attention_mask[keep_mask].unsqueeze(0)

        if self.use_speculative_decoding and self.kangaroo_model is not None:
            # ---- Run speculative decoding ----
            reply_text, self.past_key_values, spec_stats = speculative_generate_for_streaming(
                model=self.kangaroo_model,
                inputs=inputs,
                processor=self.processor,
                past_key_values=self.past_key_values,
                max_new_tokens=512,
                early_exit_layer=self.exit_layer,
                speculative_steps=self.speculative_steps,
                threshold=self.speculative_threshold,
                block_verify=(self.verify_mode == 'fast'),
            )

            combined_stats = {'speculative': spec_stats}
            if debug_print:
                print(f"[Speculative] avg_accept={spec_stats['avg_accept_length']:.2f}, "
                      f"rounds={spec_stats['total_rounds']}, tokens={spec_stats['total_tokens']}, "
                      f"tok/s={spec_stats['tokens_per_second']:.1f}, "
                      f"decode_tok/s={spec_stats['decode_tokens_per_second']:.1f}, "
                      f"total={spec_stats['total_time']:.3f}s")

            # ---- Also run autoregressive baseline for comparison ----
            if self.compare_with_baseline:
                ar_text, self.past_key_values_ar, ar_stats = ar_generate_for_streaming(
                    model=self.kangaroo_model,
                    inputs=inputs,
                    processor=self.processor,
                    past_key_values=self.past_key_values_ar,
                    max_new_tokens=512,
                )
                combined_stats['autoregressive'] = ar_stats

                speedup_total = ar_stats['total_time'] / spec_stats['total_time'] if spec_stats['total_time'] > 0 else 0
                speedup_decode = ar_stats['decode_time'] / spec_stats['decode_time'] if spec_stats.get('decode_time', 0) > 0 else 0
                spec_tokens = spec_stats['total_tokens']
                ar_tokens = ar_stats['total_tokens']
                output_match = (reply_text == ar_text)
                length_match = (spec_tokens == ar_tokens)
                combined_stats['output_match'] = output_match
                combined_stats['length_match'] = length_match
                combined_stats['speedup_ratio'] = speedup_total
                combined_stats['speedup_decode'] = speedup_decode

                if debug_print or not output_match:
                    tag = "MATCH" if output_match else "MISMATCH"
                    print(f"[AR Baseline] tokens={ar_tokens}, tok/s={ar_stats['tokens_per_second']:.1f}, time={ar_stats['total_time']:.3f}s")
                    print(f"[Compare] [{tag}] spec_tokens={spec_tokens}, ar_tokens={ar_tokens}, "
                          f"length_match={length_match}, speedup(total)={speedup_total:.2f}x, speedup(decode)={speedup_decode:.2f}x")
                    if not output_match:
                        print(f"  Spec output: {repr(reply_text[:200])}")
                        print(f"  AR   output: {repr(ar_text[:200])}")

                # Restore kangaroo model state for next speculative turn
                self.kangaroo_model.base_model.past_key_values = self.past_key_values

            self.generation_stats.append(combined_stats)
        else:
            # Standard autoregressive generation
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_start = time.perf_counter()

            model_output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                drop_method='none', drop_threshold=1.0, drop_absolute=True,      # no dropping
                do_sample=self.do_sample, temperature=self.temperature, top_k=self.top_k
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            gen_time = time.perf_counter() - t_start

            self.past_key_values = model_output.past_key_values
            output_token_ids = model_output.sequences
            output_token_ids = output_token_ids[:, inputs.input_ids.size(1):]
            num_new_tokens = output_token_ids.shape[1]
            reply_text = self.processor.batch_decode(output_token_ids, skip_special_tokens=True)[0]

            baseline_stats = {
                'autoregressive': {
                    'total_tokens': num_new_tokens,
                    'total_time': gen_time,
                    'tokens_per_second': num_new_tokens / gen_time if gen_time > 0 else 0,
                },
            }
            self.generation_stats.append(baseline_stats)
            if debug_print:
                print(f"[AR] tokens={num_new_tokens}, "
                      f"tok/s={baseline_stats['autoregressive']['tokens_per_second']:.1f}, "
                      f"time={gen_time:.3f}s")
        if query.get('must_reply', False):
            reply_text = self.must_reply_prompt + reply_text
        self.history.append({'role': 'assistant', 'content': reply_text, 'time': self.video_time})

        if debug_print: 
            print("kvcache length now:", self.past_key_values.get_seq_length())

    def inference(self, debug_print=False):
        while self.query_queue:
            self._encode_query(debug_print=debug_print)
        return {
            'conversation': copy.deepcopy(self.history),
            'drop_ratio': copy.deepcopy(self.model.model.all_drop_ratios),
            'generation_stats': copy.deepcopy(self.generation_stats),
        }


class DoNothingDataCollator:
    def __call__(self, batch):
        # Since batch size is 1, just return the first (and only) element
        return batch[0]


def round_numbers(data, n):
    if isinstance(data, list):
        return [round_numbers(d, n) for d in data]
    elif isinstance(data, dict):
        return {k: round_numbers(v, n) for k, v in data.items()}
    elif isinstance(data, float):
        return round(data, n)
    return data


def post_process_conversation_for_print(conversation):
    no_reply_text= "NO REPLY"
    # remove images from user turns
    new_conversation = list()
    for turn in conversation:
        if isinstance(turn['content'], list):
            res = ''
            for content in turn['content']:
                if 'text' in content:
                    res += content['text'].strip()
            turn['content'] = res
        if turn['role'] == 'assistant':
            if turn['content'] != no_reply_text:
                new_conversation.append(turn)
        elif turn['role'] == 'user':
            if turn['content']:
                new_conversation.append(turn)
    return new_conversation


def main():
    args = get_args()
    print(args)
    data_list = json.load(open(args.test_fname))
    args.end_idx = len(data_list) if args.end_idx is None else args.end_idx

    existing_question_ids = set()
    if os.path.exists(args.output_fname):
        for line in open(args.output_fname):
            existing_question_ids.add(json.loads(line)['question_id'])
        print(f"found {len(existing_question_ids)} existing question ids in {args.output_fname}")

    f_out = open(args.output_fname, 'a')
    wrapper = ProactiveInferenceClient(args)

    if '2_sec_per_frame' in args.test_fname:
        print("setting 2_sec_per_frame for testing, parsed from test_fname")
        frame_interval = 2
    elif '1_sec_per_frame' in args.test_fname:
        print("setting 1_sec_per_frame for testing, parsed from test_fname")
        frame_interval = 1
    elif '0.5_sec_per_frame' in args.test_fname:
        print("setting 0.5_sec_per_frame for testing, parsed from test_fname")
        frame_interval = 0.5
    else:
        raise ValueError(f"unknown fps setting for {args.test_fname}")
    print(f"setting {frame_interval=} for testing on {args.test_fname=}")
    wrapper.set_fps(frame_interval=frame_interval)

    for example_i, example in enumerate(tqdm(data_list)):
        if example['question_id'] in existing_question_ids:
            print(f"question {example['question_id']} already exists in {args.output_fname}, skip")
            continue
        if example_i < args.start_idx: continue
        if example_i >= args.end_idx: break
        wrapper.reset()
        wrapper.input_query_stream(example['conversation'])
        model_outputs = wrapper.inference(debug_print=(example_i - args.start_idx) < 2)
        res = {
            'question_id': example['question_id'],
            # 'model_response_list': [turn for turn in model_outputs['conversation'] if turn['role'] == 'assistant'],
            'model_response_list': post_process_conversation_for_print(model_outputs['conversation']),   # keep the user turns for easier human inspection of data quality
            'drop_ratio_list': model_outputs['drop_ratio'],
            'generation_stats': model_outputs['generation_stats'],
        }
        f_out.write(json.dumps(res) + '\n')
        f_out.flush()

        # Collect per-example speed stats
        for s in model_outputs['generation_stats']:
            all_stats.append(s)

    f_out.close()

    # Print aggregate speed metrics
    if all_stats:
        print("\n" + "=" * 60)
        print(f"Aggregate Speed Metrics ({len(all_stats)} generations)")
        print("=" * 60)

        # Autoregressive stats (always present)
        ar_stats_list = [s['autoregressive'] for s in all_stats if 'autoregressive' in s]
        if ar_stats_list:
            ar_total_tokens = sum(s['total_tokens'] for s in ar_stats_list)
            ar_total_time = sum(s['total_time'] for s in ar_stats_list)
            ar_tok_s = ar_total_tokens / ar_total_time if ar_total_time > 0 else 0
            print(f"[Autoregressive]")
            print(f"  Total tokens:    {ar_total_tokens}")
            print(f"  Total time:      {ar_total_time:.2f}s")
            print(f"  Tokens/sec:      {ar_tok_s:.1f}")

        # Speculative stats
        spec_stats_list = [s['speculative'] for s in all_stats if 'speculative' in s]
        if spec_stats_list:
            spec_total_tokens = sum(s['total_tokens'] for s in spec_stats_list)
            spec_total_time = sum(s['total_time'] for s in spec_stats_list)
            spec_tok_s = spec_total_tokens / spec_total_time if spec_total_time > 0 else 0
            accept_lengths = []
            total_decode_time = 0
            for s in spec_stats_list:
                accept_lengths.extend(s.get('accept_lengths', []))
                total_decode_time += s.get('decode_time', 0)
            decode_tok_s = spec_total_tokens / total_decode_time if total_decode_time > 0 else 0
            avg_accept = sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0

            print(f"[Speculative Decoding]")
            print(f"  Total tokens:    {spec_total_tokens}")
            print(f"  Total time:      {spec_total_time:.2f}s")
            print(f"  Tokens/sec:      {spec_tok_s:.1f}")
            print(f"  Decode tok/s:    {decode_tok_s:.1f}")
            print(f"  Avg accept len:  {avg_accept:.2f}")
            print(f"  Total rounds:    {sum(s['total_rounds'] for s in spec_stats_list)}")

        # Comparison (when both are available)
        if ar_stats_list and spec_stats_list:
            speedup_ratios = [s['speedup_ratio'] for s in all_stats if 'speedup_ratio' in s]
            speedup_decode_ratios = [s['speedup_decode'] for s in all_stats if 'speedup_decode' in s]
            matches = [s['output_match'] for s in all_stats if 'output_match' in s]
            length_matches = [s['length_match'] for s in all_stats if 'length_match' in s]
            avg_speedup = sum(speedup_ratios) / len(speedup_ratios) if speedup_ratios else 0
            avg_speedup_decode = sum(speedup_decode_ratios) / len(speedup_decode_ratios) if speedup_decode_ratios else 0
            match_rate = sum(matches) / len(matches) if matches else 0
            length_match_rate = sum(length_matches) / len(length_matches) if length_matches else 0
            overall_speedup = ar_total_time / spec_total_time if spec_total_time > 0 else 0

            # Decode-only overall speedup
            ar_total_decode = sum(s.get('decode_time', s['total_time']) for s in ar_stats_list)
            spec_total_decode = sum(s.get('decode_time', s['total_time']) for s in spec_stats_list)
            overall_decode_speedup = ar_total_decode / spec_total_decode if spec_total_decode > 0 else 0

            print(f"[Comparison]")
            print(f"  Verify mode:             {wrapper.verify_mode}")
            print(f"  Overall speedup (total):  {overall_speedup:.2f}x")
            print(f"  Overall speedup (decode): {overall_decode_speedup:.2f}x")
            print(f"  Avg per-turn speedup (total):  {avg_speedup:.2f}x")
            print(f"  Avg per-turn speedup (decode): {avg_speedup_decode:.2f}x")
            print(f"  Output match rate:    {match_rate:.1%} ({sum(matches)}/{len(matches)})")
            print(f"  Length match rate:    {length_match_rate:.1%} ({sum(length_matches)}/{len(length_matches)})")
            if match_rate < 1.0:
                num_mismatch = len(matches) - sum(matches)
                print(f"  WARNING: {num_mismatch} output(s) differ between AR and speculative!")
        print("=" * 60)


if __name__ == '__main__':
    all_stats = []
    main()