# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: tokenizer for SVG data, wrapping a base HuggingFace tokenizer

import re
from functools import partial
import logging  # Use standard logging for warnings/errors
from typing import List, Dict, Any, Optional, Union, Tuple  # Added more types

import omegaconf
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # Import base class
import torch

from .semantic_tokens import SVGToken, AttribMapper, ContainerMapper, GradientsMapper, PathCMDMapper, PathMapper, \
    ShapeMapper, syntactic2svg
from .token_config import SEMANTIC_SVG_TOKEN_MAPPER_DEFAULT as TokenDescMapper

logger = logging.getLogger(__name__)  # Use standard logger for internal messages

ALLMapper = {
    **PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper, **GradientsMapper, **AttribMapper
}
SVG_TOKENS = list(ALLMapper.values())
SVG_SYMBOLS = list(SVGToken.values())
# Common representation for space in tokenizers like GPT-2
SPACE_TOKEN = "Ġ"

NUM_TOKEN = "[<|COORD|>]"
COORD_TOKEN = ["[<|NUM_X|>]", "[<|NUM_Y|>]"]
COLOR_TOKEN = "[<|COLOR|>]"
TEXT_SYMBOL = ['[<|START_OF_TEXT|>]', '[<|END_OF_TEXT|>]']
IMG_START_TOKEN = '[<|START_OF_IMG|>]'
IMG_TOKEN = "[<|IMG|>]"
IMG_END_TOKEN = '[<|END_OF_IMG|>]'
MASK_TOKEN = "[<|MASK|>]"

# Standard ignore index for loss functions
IGNORE_INDEX = -100

# test case (keep for reference or move to tests)
SVG_DESC_T1 = "[<|START_OF_SVG|>][<|svg_path|>][<|d|>][<|moveto|>]63.6 118.8[<|curveto|>]-27.9 0 -58 -17.5 -58 -55.9[<|smooth curveto|>]30.1 -55.9 58 -55.9[<|curveto|>]-10.6 9.3 -25 14.5 -40.4 14.5[<|close the path|>][<|fill|>]#fde030[<|END_OF_SVG|>]"


class SVGTokenizer:
    """
    A tokenizer for SVG data that wraps a Hugging Face AutoTokenizer,
    adding custom SVG tokens and special tokens based on configuration.

    Provides standard tokenization methods and specialized methods for handling
    numerical values within SVG descriptions (use `tokenize_with_num` with caution).
    """

    def __init__(self, cfg: omegaconf.DictConfig, print_fn=partial(print), **kwargs):
        """
        Initializes the SVGTokenizer.

        Args:
            cfg (omegaconf.DictConfig): Configuration object containing settings like
                                        tokenizer_name, num_token, coord_token, etc.
            print_fn (callable): Function used for printing initialization info.
            **kwargs: Additional keyword arguments passed to AutoTokenizer.from_pretrained.
        """
        self.print_fn = print_fn
        self.cfg = cfg
        self.init_kwargs = kwargs

        # Load base tokenizer
        try:
            base_tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                cfg.tokenizer_name, trust_remote_code=True, **kwargs
            )
            self.print_fn(f"Loaded base tokenizer '{cfg.tokenizer_name}'")
        except Exception as e:
            logger.error(f"Failed to load base tokenizer '{cfg.tokenizer_name}': {e}", exc_info=True)
            raise e

        self.original_vocab_size = len(base_tokenizer)
        self.print_fn(f"Original vocab size: {self.original_vocab_size}")

        # Prepare custom tokens to add
        tokens_to_add = [t for t in SVG_TOKENS if t not in SVG_SYMBOLS]  # Non-special SVG tokens

        self.use_num_token = cfg.get('num_token', False)
        if self.use_num_token:
            tokens_to_add.append(NUM_TOKEN)
            TokenDescMapper[NUM_TOKEN] = 'number, the value of the coordinate'
        if cfg.get('coord_token', False):
            tokens_to_add.extend(COORD_TOKEN)  # Use extend for list
            TokenDescMapper[COORD_TOKEN[0]] = 'number, the value of the x coordinate'
            TokenDescMapper[COORD_TOKEN[1]] = 'number, the value of the y coordinate'
        if cfg.get('rgb_token', False):
            tokens_to_add.append(COLOR_TOKEN)
            TokenDescMapper[COLOR_TOKEN] = "RGB color value"
        if cfg.get('mask_token', False):
            tokens_to_add.append(MASK_TOKEN)
            TokenDescMapper[MASK_TOKEN] = "mask"

        # Add non-special tokens
        num_added_tokens = base_tokenizer.add_tokens(tokens_to_add)
        if num_added_tokens > 0:
            self.print_fn(f"Added {num_added_tokens} new non-special SVG tokens.")

        # Prepare special tokens
        special_tokens_to_add = list(SVG_SYMBOLS)  # Start with base SVG symbols
        if cfg.get('add_txt_token', False):
            special_tokens_to_add.extend(TEXT_SYMBOL)
        if cfg.get('add_img_token', False):
            special_tokens_to_add.extend([IMG_START_TOKEN, IMG_TOKEN, IMG_END_TOKEN])

        # Ensure uniqueness and sort
        special_tokens_to_add = sorted(list(set(special_tokens_to_add)))

        # Add special tokens dictionary
        special_tokens_dict = {'additional_special_tokens': special_tokens_to_add}
        num_added_special_tokens = base_tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_special_tokens > 0:
            self.print_fn(f"Added {num_added_special_tokens} new special tokens: {special_tokens_to_add}")

        # Store the list of *all* custom tokens (special or not) that were intended
        self._intended_custom_tokens = sorted(list(set(tokens_to_add + special_tokens_to_add)))
        # Store the list of tokens that were actually new after adding operations
        try:
            self._newly_added_tokens = sorted(list(base_tokenizer.get_added_vocab().keys()))
        except AttributeError:
            logger.warning(
                "Cannot get exact list of newly added tokens (requires transformers >= 4.3). Relying on initial check.")
            # Fallback based on initial check (less accurate if base tokenizer already had some)
            _new_non_special = set(tokens_to_add) - set(base_tokenizer.vocab.keys())
            _new_special = set(special_tokens_to_add) - set(base_tokenizer.get_vocab(with_added_tokens=False).keys())
            self._newly_added_tokens = sorted(list(_new_non_special | _new_special))  # Approximate

        # Process pad token
        pad_token_type = cfg.get('pad_token', 'eos_token').lower()  # Default to eos_token
        if pad_token_type == '[<|pad|>]':  # Normalize token string
            num_added = base_tokenizer.add_special_tokens({'pad_token': '[<|PAD|>]'})
            if num_added > 0: self._newly_added_tokens.append('[<|PAD|>]')
        elif pad_token_type == '[pad]':
            num_added = base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if num_added > 0: self._newly_added_tokens.append('[PAD]')
        elif pad_token_type == 'eos_token':
            if base_tokenizer.eos_token:
                base_tokenizer.pad_token = base_tokenizer.eos_token
                self.print_fn(f"Set pad_token to eos_token: {base_tokenizer.eos_token}")
            else:
                logger.warning("pad_token='eos_token' but eos_token is not set. Adding default [PAD].")
                num_added = base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                if num_added > 0: self._newly_added_tokens.append('[PAD]')
        elif base_tokenizer.pad_token is None:
            logger.warning(
                f"pad_token_type '{pad_token_type}' not recognized or base tokenizer has no default pad token. Adding default [PAD].")
            num_added = base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if num_added > 0: self._newly_added_tokens.append('[PAD]')

        self.print_fn(f"Final vocab size: {len(base_tokenizer)}")
        if self.original_vocab_size != len(base_tokenizer):
            self.print_fn(f"Total tokens added: {len(base_tokenizer) - self.original_vocab_size}")

        self.tokenizer = base_tokenizer  # Store the final tokenizer instance
        self.vocab_size = len(self.tokenizer)
        self.max_length = cfg.seq_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.padding_side = self.tokenizer.padding_side
        self.ignore_id = IGNORE_INDEX  # Use constant

    def get_new_tokens(self) -> List[str]:
        """Returns a list of token strings that were newly added to the base tokenizer."""
        return sorted(list(set(self._newly_added_tokens)))

    def encode_tokens(self, text: str) -> List[str]:
        """Encodes text into a list of token strings."""
        return self.tokenizer.tokenize(text, add_special_tokens=True)

    def __call__(self, text: Union[str, List[str]], max_length: Optional[int] = None, padding: Union[bool, str] = True,
                 truncation: bool = True, return_tensors: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Tokenizes text using the underlying Hugging Face tokenizer."""
        return self.tokenizer(
            text,
            add_special_tokens=True,  # Typically True for model input
            max_length=max_length if max_length is not None else self.max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs  # Pass additional kwargs
        )

    # Kept tokenize for backward compatibility if used elsewhere explicitly
    def tokenize(self, text: Union[str, List[str]], max_length: Optional[int] = None, padding: Union[bool, str] = True,
                 truncation: bool = True, return_tensors: Optional[str] = None, **kwargs):
        """Alias for __call__."""
        return self(text, max_length, padding, truncation, return_tensors, **kwargs)

    def tokenize_ids(self, text: Union[str, List[str]], max_length: Optional[int] = None,
                     padding: Union[bool, str] = True, truncation: bool = True, return_tensors: Optional[str] = None) -> \
            Union[List[int], List[List[int]], Any]:
        """Tokenizes text and returns only the input_ids."""
        results = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length if max_length is not None else self.max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )
        return results['input_ids']

    def tokens2ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Converts token strings to their corresponding IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_space_token_id(self) -> Optional[int]:
        """Gets the ID for the space token (implementation specific, e.g., 'Ġ')."""
        try:
            return self.tokens2ids(SPACE_TOKEN)
        except KeyError:
            logger.warning(f"Space token '{SPACE_TOKEN}' not found in tokenizer vocab.")
            return None

    # Get specific token IDs
    def _get_token_id_safe(self, token: str) -> Optional[int]:
        """Helper to safely get token ID, returns None if not found."""
        try:
            return self.tokens2ids(token)
        except KeyError:
            logger.warning(f"Token '{token}' not found in tokenizer vocab.")
            return None

    def get_number_token_id(self) -> Optional[int]:
        return self._get_token_id_safe(NUM_TOKEN)

    def get_color_token_id(self) -> Optional[int]:
        return self._get_token_id_safe(COLOR_TOKEN)

    def get_mask_token_id(self) -> Optional[int]:
        return self._get_token_id_safe(MASK_TOKEN)

    def get_vision_start_id(self) -> Optional[int]:
        return self._get_token_id_safe(IMG_START_TOKEN)

    def get_vision_end_id(self) -> Optional[int]:
        return self._get_token_id_safe(IMG_END_TOKEN)

    # Specialized Tokenization (Use with Caution)

    def tokenize_with_num(
            self,
            text: str,
            ignore_val: int = -10000,
            with_colors: bool = False,
            max_length: Optional[int] = None,
            padding: Union[bool, str] = True,
            truncation: bool = True,
            return_tensors: Optional[str] = None
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        **Experimental:** Tokenizes text, replacing numbers/colors with special tokens
        and returning separate tensors for IDs and numerical values.

        **Warning:** This method modifies the text significantly using regex and
        removes space tokens, which may be incompatible with standard transformer models.
        Use only if the downstream model architecture specifically requires this format.

        Args:
            text (str): Input text string.
            ignore_val (int): Value used in num_embed for non-number locations.
            with_colors (bool): Whether to separate colors into color_embed.
            max_length, padding, truncation, return_tensors: Passed to tokenizer.

        Returns:
            Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
            A tuple containing (ids, num_embed, color_embed), or None if an error occurs.
            ids: Tensor of token IDs with placeholders.
            num_embed: Tensor of the same shape as ids, containing number values at
                       NUM_TOKEN locations and ignore_val elsewhere.
            color_embed: Tensor for color values (if with_colors=True), otherwise None.
        """
        logger.warning("Method 'tokenize_with_num' is experimental and removes space tokens, "
                       "which may break standard model compatibility. Use with caution.")
        if not self.use_num_token:
            logger.error("Cannot use tokenize_with_num: 'num_token' was set to False during initialization.")
            return None

        num_token_id = self.get_number_token_id()
        if num_token_id is None:
            logger.error("Cannot use tokenize_with_num: NUM_TOKEN ID not found.")
            return None

        space_token_id = self.get_space_token_id()
        # Ensure space token ID is available if we plan to remove it
        if space_token_id is None:
            logger.error("Cannot use tokenize_with_num: Space token ID ('Ġ') not found.")
            return None

        color_token_id = self.get_color_token_id() if with_colors else None
        if with_colors and color_token_id is None:
            logger.error("Cannot use tokenize_with_num with with_colors=True: COLOR_TOKEN ID not found.")
            return None

        processed_text = text.replace(']', '] ').replace('[', ' [')  # Add spaces around brackets
        processed_text = re.sub(r'\s+', ' ', processed_text).strip()  # Normalize spaces

        def _parse_number(num_str: str, round_num: int = 2) -> Optional[Union[int, float]]:
            try:
                num = float(num_str)
                return int(num) if num.is_integer() else round(num, round_num)
            except ValueError:
                logger.warning(f"Could not parse number: '{num_str}'")
                return None  # Return None on failure

        # Extract colors and numbers carefully
        extracted_colors = []
        text_no_colors = processed_text
        if with_colors:
            extracted_colors = re.findall(r'#(?:[0-9a-fA-F]{3}){1,2}', processed_text)
            text_no_colors = re.sub(r'#(?:[0-9a-fA-F]{3}){1,2}', f" {COLOR_TOKEN} ",
                                    processed_text)  # Add spaces around token

        # Use findall to get numbers in order
        number_strings = re.findall(r'-?\d*\.?\d+', text_no_colors)
        extracted_numbers = [_parse_number(n) for n in number_strings]
        extracted_numbers = [n for n in extracted_numbers if n is not None]  # Filter out parsing errors

        # Replace numbers with placeholder, adding spaces
        text_placeholders = re.sub(r'-?\d*\.?\d+', f" {NUM_TOKEN} ", text_no_colors)
        # Normalize spaces again after replacements
        text_placeholders = re.sub(r'\s+', ' ', text_placeholders).strip()

        # Tokenize the text with placeholders
        try:
            # Tokenize single text, expect list of lists/tensors if return_tensors is set
            tokenized_output = self.tokenizer(
                [text_placeholders],
                add_special_tokens=True,
                max_length=max_length if max_length is not None else self.max_length,
                truncation=truncation,
                padding=padding,
                return_tensors='pt'
            )
            ids_tensor = tokenized_output['input_ids'][0]
            # Keep mask if needed later
            # attention_mask = tokenized_output['attention_mask'][0]
        except Exception as e:
            logger.error(f"Error during tokenization of placeholder text: {e}", exc_info=True)
            return None

        # This logic is generally discouraged for standard transformers.
        original_len = len(ids_tensor)
        ids_no_space = ids_tensor[ids_tensor != space_token_id]
        if len(ids_no_space) != original_len:
            logger.warning(
                f"Removed {original_len - len(ids_no_space)} space tokens. This might affect model performance.")

        ids_final = ids_no_space

        # Create number embedding tensor
        num_locs = (ids_final == num_token_id)
        num_embed = torch.full_like(ids_final, fill_value=ignore_val, dtype=torch.float)
        num_count_in_ids = num_locs.sum().item()

        if num_count_in_ids != len(extracted_numbers):
            logger.warning(f"Mismatch between NUM_TOKEN count ({num_count_in_ids}) "
                           f"and extracted numbers count ({len(extracted_numbers)}). "
                           f"Number embedding may be incorrect.")
            # Truncate or pad numbers list? Truncate for now.
            numbers_to_assign = extracted_numbers[:num_count_in_ids]
        else:
            numbers_to_assign = extracted_numbers

        if numbers_to_assign:
            num_embed[num_locs] = torch.tensor(numbers_to_assign, dtype=torch.float)

        # Create color embedding tensor (if requested)
        color_embed = None
        if with_colors:
            color_locs = (ids_final == color_token_id)
            color_embed = torch.full_like(ids_final, fill_value=ignore_val,
                                          dtype=torch.float)  # Use ignore val? Or 0/1?
            color_count_in_ids = color_locs.sum().item()

            if color_count_in_ids != len(extracted_colors):
                logger.warning(f"Mismatch between COLOR_TOKEN count ({color_count_in_ids}) "
                               f"and extracted colors count ({len(extracted_colors)}). "
                               f"Color embedding may be incorrect.")
                # Need to decide how to handle color values (e.g., parse hex to RGB floats?)
                # Placeholder: Assigning index for now
                colors_to_assign = list(range(color_count_in_ids))  # Simple index, NOT real colors
            else:
                # Placeholder: Assigning index for now
                colors_to_assign = list(range(color_count_in_ids))  # Simple index, NOT real colors

            if colors_to_assign:
                # Assign placeholder value (e.g., index)
                color_embed[color_locs] = torch.tensor(colors_to_assign, dtype=torch.float)

        # Return based on return_tensors requested by caller (currently forced 'pt')
        # Adapt if non-tensor output is needed
        return ids_final, num_embed, color_embed

    # Decoding Methods
    def decode(self, token_ids: Union[int, List[int], Any], skip_special_tokens: bool = False, **kwargs) -> str:
        """Decodes token IDs back into a string."""
        # Handle tensor input
        if hasattr(token_ids, "tolist"):  # Check if it's tensor-like
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    # Renamed decode_token to decode_ids for clarity
    def decode_ids(self, token_ids: Union[int, List[int], Any], skip_special_tokens: bool = False, **kwargs) -> str:
        """Alias for decode."""
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode_syntactic(self, svg_desc: str) -> str:
        """Converts a syntactic SVG description string into standard SVG format."""
        try:
            decoded = syntactic2svg(svg_desc)
            return decoded
        except Exception as e:
            logger.error(f"Error during syntactic SVG decoding: {e}", exc_info=True)
            return f"<error>Decoding failed: {e}</error>"  # Return error string

    def decode2svg(self, token_ids: Union[List[int], Any]) -> str:
        """Decodes token IDs into a syntactic SVG description and then converts to SVG format."""
        svg_desc = self.decode(token_ids, skip_special_tokens=True)  # Skip special tokens for cleaner syntax
        decoded_svg = self.decode_syntactic(svg_desc)
        return decoded_svg

    def decode_with_num(self, token_ids: torch.Tensor, num_preds: torch.Tensor) -> str:
        """
        **Experimental:** Decodes token IDs, replacing NUM_TOKEN placeholders with predicted numbers.

        **Warning:** Relies on the experimental `tokenize_with_num` logic and may produce
        incorrect results if token/number alignment was lost. Space token removal is NOT reversed.
        """
        logger.warning("Method 'decode_with_num' is experimental and corresponds to 'tokenize_with_num'. "
                       "It does not re-insert removed space tokens.")
        if not self.use_num_token:
            logger.error("Cannot use decode_with_num: 'num_token' was False during initialization.")
            return "<error> num_token not configured </error>"

        num_token_id = self.get_number_token_id()
        if num_token_id is None:
            logger.error("Cannot use decode_with_num: NUM_TOKEN ID not found.")
            return "<error> NUM_TOKEN not found </error>"

        # Decode the raw token IDs first (potentially including NUM_TOKEN placeholders)
        text = self.decode(token_ids, skip_special_tokens=False)  # Keep placeholders for replacement

        # Extract numbers corresponding to NUM_TOKEN positions in the *original* id tensor
        num_mask = (token_ids == num_token_id)
        # Filter the predicted numbers based on the mask, excluding the ignore_id value
        filtered_numbers = num_preds[num_mask & (num_preds != self.ignore_id)]
        numbers_to_insert = [f"{num.item():.2f}" for num in filtered_numbers]  # Format numbers

        # Replace NUM_TOKEN placeholders in the decoded text
        # Use a way to avoid replacing already replaced tokens if NUM_TOKEN string occurs naturally
        # This regex replacement is still fragile if NUM_TOKEN appears multiple times consecutively etc.
        def _replacer(match):
            if numbers_to_insert:
                # Add space around number for basic separation
                return f" {numbers_to_insert.pop(0)} "
            else:
                logger.warning("More NUM_TOKEN placeholders found than numbers available for insertion.")
                return " <missing_num> "  # Placeholder for missing number

        # Escape NUM_TOKEN in case it contains regex special characters
        escaped_token = re.escape(NUM_TOKEN)
        reconstructed_text = re.sub(escaped_token, _replacer, text)

        if numbers_to_insert:  # Check if any numbers were left over
            logger.warning(f"Numbers left over after replacement: {numbers_to_insert}")

        # Clean up potential double spaces resulting from replacement
        reconstructed_text = re.sub(r'\s+', ' ', reconstructed_text).strip()

        return reconstructed_text

    def apply_chat_template(self, *args, **kwargs):
        """Applies the chat template defined in the underlying tokenizer."""
        if not hasattr(self.tokenizer, 'apply_chat_template'):
            logger.error("The underlying tokenizer does not support apply_chat_template.")
            # Fallback or raise error
            raise NotImplementedError("apply_chat_template not available.")
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def test(self, sentence: str):
        """Runs simple encoding/decoding tests and prints results."""
        self.print_fn("-" * 50)
        self.print_fn(f"Testing sentence: '{sentence}'")
        try:
            encoded_tokens = self.encode_tokens(sentence)
            self.print_fn("Encoded Tokens: ", encoded_tokens)

            encoded_call = self(sentence, padding=False, truncation=False)  # No padding/trunc
            self.print_fn("Tokenize (__call__): ", encoded_call)
            input_ids = encoded_call['input_ids']
            # Ensure input_ids is a list for len()
            input_ids_list = input_ids[0] if isinstance(input_ids, list) and len(input_ids) == 1 else input_ids
            self.print_fn("Tokenize len: ", len(input_ids_list))

            # Use decode method which handles tensors
            decoded = self.decode(input_ids, skip_special_tokens=False)
            self.print_fn("Decoded: ", decoded)
        except Exception as e:
            logger.error(f"Error during test: {e}", exc_info=True)
        self.print_fn("-" * 50)

    def __len__(self) -> int:
        """Returns the current vocabulary size."""
        # The vocab size is fixed after __init__, no need to recalculate
        return self.vocab_size

    def save_pretrained(self, save_directory: str, **kwargs):
        """Saves the underlying tokenizer to a directory."""
        return self.tokenizer.save_pretrained(save_directory, **kwargs)
