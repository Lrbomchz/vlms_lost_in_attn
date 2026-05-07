from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.models.auto import AutoModel
from SAP_models.my_modeling_llama import LlamaModel as MyLlamaModel

from transformers.models.llava.configuration_llava import LlavaConfig
from utils.SAP_utils import Apipe_prob


logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava outputs, with hidden states and attentions.
    """
)
class LlavaModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None
    vision_attentions: Optional[tuple[torch.FloatTensor]] = None  # ✅ 新增


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for Llava causal language model (or autoregressive) outputs.
    """
)
class LlavaCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    vision_attentions: Optional[tuple[torch.FloatTensor]] = None  # ✅ 新增


class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


@auto_docstring
class LlavaPreTrainedModel(PreTrainedModel):
    config: LlavaConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@auto_docstring(
    custom_intro="""
    The Llava model which consists of a vision backbone and a language model, without a language modeling head.
    """
)
class LlavaModel(LlavaPreTrainedModel):
    _checkpoint_conversion_mapping = {"language_model.model": "language_model"}

    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = LlavaMultiModalProjector(config)
        #self.language_model = AutoModel.from_config(config.text_config)
        self.language_model = MyLlamaModel(config.text_config)

        # ====== 运行时可改的配置（对齐你 Qwen）======
        self.visual_token_range: Optional[tuple[int, int]] = None
        self.visual_align_lambda: float = 0.5
        self.visual_head_percentile_min: float = 0.0
        self.visual_head_percentile_max: float = 1.0
        self.apipe_mode: str = "vis_enc_attn"
        self.vis_weight = None
        self.vision_attentions = None  # 用来存 vision tower 的 attentions

        self.cached_aligned_prob = None  # (num_patch,) 仅缓存这个
        self.cached_visual_token_range = None

        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def set_visual_guidance_config(
            self,
            visual_token_range: tuple[int, int],
            apipe_mode: str = "vis_enc_attn",
            align_lambda: float = 0.5,
            head_percentile_min: float = 0.0,
            head_percentile_max: float = 1.0,
            vis_weight=None,
            replace_layer="all",
    ):
        self.visual_token_range = visual_token_range
        self.visual_align_lambda = align_lambda
        self.visual_head_percentile_min = head_percentile_min
        self.visual_head_percentile_max = head_percentile_max
        self.apipe_mode = apipe_mode
        self.vis_weight = vis_weight
        self.replace_layer = replace_layer

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        output_vision_attentions: bool = False,  # ✅ 新增
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`):
               The tensors corresponding to the input images.
            vision_feature_layer (`Union[int, list[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden states.
        image_outputs = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
            output_attentions=output_vision_attentions,  # ✅ 新增
            **kwargs,
        )

        if output_vision_attentions and getattr(image_outputs, "attentions", None) is not None:
            # 只保留最后一层，避免整套 tuple 常驻
            self.vision_attentions = (image_outputs.attentions[-1],)
        else:
            self.vision_attentions = None

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        image_features = self.multi_modal_projector(selected_image_feature)

        if "image_sizes" in kwargs:
            split_sizes = [
                (height // self.vision_tower.patch_size) * (width // self.vision_tower.patch_size)
                for height, width in kwargs["image_sizes"]
            ]
            image_features = torch.split(image_features.squeeze(0), split_sizes)
        else:
            image_features = list(image_features)
        return image_features

    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )
        return special_image_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,  # ✅ 新增
        output_hidden_states: Optional[bool] = None,  # ✅ 新增
        return_dict: Optional[bool] = None,  # ✅ 新增
        output_vision_attentions: Optional[bool] = None,  # ✅ 新增
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaModelOutputWithPast]:

        # NEW INIT
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_vision_attentions = bool(output_vision_attentions) if output_vision_attentions is not None else False

        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        is_decode = (past_key_values is not None) and (cache_position is not None) and (cache_position[0].item() > 0)

        # ====== 只在 prefill 清空；decode 需要复用 cached guidance ======
        if (not is_decode) and hasattr(self.language_model, "layers"):
            for layer in self.language_model.layers:
                if hasattr(layer, "self_attn"):
                    sa = layer.self_attn
                    if hasattr(sa, "visual_guidance"):
                        sa.visual_guidance = None
                        sa.visual_token_range = None
                # 运行时参数同步可保留（不影响复用）
                if hasattr(layer, "self_attn"):
                    sa = layer.self_attn
                    sa.visual_lambda = getattr(self, "visual_align_lambda", 0.5)
                    sa.visual_head_percentile_min = getattr(self, "visual_head_percentile_min", 0.0)
                    sa.visual_head_percentile_max = getattr(self, "visual_head_percentile_max", 1.0)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            need_vision_attn = (self.apipe_mode == "vis_enc_attn")
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
                output_vision_attentions=need_vision_attn,   # aligned 才计算 attn
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        aligned_prob = None
        vtr = None

        if is_decode:
            # ===== decode：直接复用 prefill 缓存 =====
            aligned_prob = self.cached_aligned_prob
            vtr = self.cached_visual_token_range
        else:
            # ===== prefill：计算 guidance 并缓存 =====
            vtr = self.visual_token_range

            if pixel_values is not None:
                if self.apipe_mode == "vis_enc_attn":
                    if self.vision_attentions is None:
                        raise ValueError("vis_enc_attn mode requires vision attentions, but got None.")
                    last_attn = self.vision_attentions[0]  # (B,H,S,S)
                    patch_attn = last_attn[:, :, 1:, 1:]  # (B,H,P,P)
                    per_head_attn = patch_attn[0]  # (H,P,P)
                    vis_attn = per_head_attn.mean(dim=1).mean(dim=0)  # (P,)
                elif self.apipe_mode in ["complexity", "key"]:
                    # ✅ non-aligned：完全不需要 vision attentions
                    if self.vis_weight is None:
                        raise ValueError(f"Apipe mode {self.apipe_mode} needs vis_weight.")
                    vis_attn = self.vis_weight.to(inputs_embeds.device)
                elif self.apipe_mode == "gaussian_noise":
                    vis_attn = self.vis_weight.to(inputs_embeds.device)
                else:
                    raise ValueError(f"Unsupported Apipe mode: {self.apipe_mode}")

                aligned_prob = Apipe_prob(self.apipe_mode, vis_attn)

                # ✅ 缓存：只缓存 (P,) 并 detach，供 decode 复用
                if aligned_prob is not None:
                    self.cached_aligned_prob = aligned_prob.detach()
                    self.cached_visual_token_range = vtr

                # ✅ 如果你不打算把 vision_attentions 作为输出返回，立刻释放引用
                if not output_vision_attentions:
                    self.vision_attentions = None

        # ===== prefill & decode：每次都把 cached/计算出的 guidance 灌到 decoder =====
        if aligned_prob is not None and vtr is not None and hasattr(self.language_model, "layers"):
            for idx, layer in enumerate(self.language_model.layers):
                #if idx == 0:
                #if idx not in [12, 13, 14, 15, 16, 17, 18]: 8
                #if idx not in [18, 19, 20, 21, 22, 23, 24]: 6
                #if idx not in [24, 25, 26, 27, 28, 29, 30]: 5

                #if idx in [0, 1, 2, 3, 4, 5, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15]:
                #if idx not in [6, 7, 8, 9, 10, 11]:
                if self.replace_layer == "all":
                    if idx == 0:
                        continue
                elif str(idx) not in self.replace_layer.split(","):
                    continue
                sa = layer.self_attn
                sa.visual_guidance = aligned_prob
                sa.visual_token_range = vtr
                sa.visual_lambda = getattr(self, "visual_align_lambda", 0.5)
                sa.visual_head_percentile_min = getattr(self, "visual_head_percentile_min", 0.0)
                sa.visual_head_percentile_max = getattr(self, "visual_head_percentile_max", 1.0)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # 建议固定 True，外面再按 return_dict 处理
            **kwargs,
        )

        return LlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            vision_attentions=self.vision_attentions,
        )


@auto_docstring(
    custom_intro="""
    The LLAVA model which consists of a vision backbone and a language model.
    """
)
class LlavaForConditionalGeneration(LlavaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.model = LlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

    # Make modules available through conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_vision_attentions: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, LlavaCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            cache_position=cache_position,
            image_sizes=image_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            output_vision_attentions=output_vision_attentions,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            vision_attentions=outputs.vision_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
            # Otherwise we need pixel values to be passed to model
            model_inputs["pixel_values"] = pixel_values
            # prefill 才允许跑 vision attentions
            model_inputs["output_vision_attentions"] = kwargs.get("output_vision_attentions", False)
        else:
            model_inputs["output_vision_attentions"] = False

        return model_inputs


__all__ = ["LlavaForConditionalGeneration", "LlavaPreTrainedModel", "LlavaModel"]
