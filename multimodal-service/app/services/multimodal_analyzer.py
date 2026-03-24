from __future__ import annotations

import datetime
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import requests
import torch
from huggingface_hub import hf_hub_download
import open_clip
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

from app.core.config import (
    DEVICE,
    EYE_MODEL_FILENAME,
    EYE_MODEL_ID,
    EYE_MODEL_REVISION,
    EYE_MODEL_SUBDIR,
    EYE_RUN_CONFIG_FILENAME,
    EYE_TOP_K,
    HF_TOKEN,
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS,
    MODEL_CACHE_DIR,
    REQUEST_TIMEOUT_SECONDS,
    ROUTE_MIN_CONFIDENCE,
    ROUTE_MODEL_ID,
    ROUTE_MODEL_PRETRAINED,
    ROUTE_MODEL_REVISION,
    ROUTE_TOP_K,
    SKIN_MODEL_FILENAME,
    SKIN_MODEL_ID,
    SKIN_MODEL_REVISION,
    SKIN_MODEL_SUBDIR,
    SKIN_RUN_CONFIG_FILENAME,
    SKIN_TOP_K,
)
from app.schemas.multimodal_schema import (
    ClassScore,
    EyeDiseaseResult,
    MultimodalAnalyzeRequest,
    MultimodalAnalyzeResponse,
    RouteResult,
    SkinDiseaseResult,
)

logger = logging.getLogger(__name__)
SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _torch_load(path: str, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_local_or_hf_file(
    model_id_or_path: str,
    *,
    filename: str,
    subdir: str = "",
    revision: str | None = None,
) -> str:
    candidate = Path(model_id_or_path)
    if candidate.is_file():
        return str(candidate)
    if candidate.is_dir():
        nested = candidate / subdir / filename if subdir else candidate / filename
        if nested.exists():
            return str(nested)

    hf_filename = f"{subdir}/{filename}" if subdir else filename
    cache_dir = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=model_id_or_path,
        filename=hf_filename,
        revision=revision,
        token=HF_TOKEN,
        cache_dir=str(cache_dir),
    )


class ImageRouteClassifier:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_id = ROUTE_MODEL_ID
        if ROUTE_MODEL_REVISION:
            logger.warning("ROUTE_MODEL_REVISION is ignored when using open_clip MobileCLIP loading.")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id,
            pretrained=ROUTE_MODEL_PRETRAINED,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.prompt_groups: Dict[str, List[str]] = {
            "eye_closeup": [
                "a close-up photo of a dog's eye",
                "a macro photo focused on a dog's eye",
            ],
            "skin_closeup": [
                "a close-up photo of a dog's skin",
                "a macro photo focused on a dog's skin",
            ],
            "other": [
                "a photo that is not a close-up of a dog's eye or dog's skin",
                "an unrelated pet photo without a close-up eye or skin lesion",
            ],
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> RouteResult:
        flat_prompts: List[str] = []
        prompt_to_label: List[str] = []
        for label, prompts in self.prompt_groups.items():
            for prompt in prompts:
                flat_prompts.append(prompt)
                prompt_to_label.append(label)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(flat_prompts).to(self.device)

        image_embeds = self.model.encode_image(image_input)
        text_embeds = self.model.encode_text(text_tokens)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_embeds @ text_embeds.T

        prompt_logits = logits[0]
        grouped_logits: Dict[str, List[torch.Tensor]] = {}
        for index, label in enumerate(prompt_to_label):
            grouped_logits.setdefault(label, []).append(prompt_logits[index])

        labels = list(self.prompt_groups.keys())
        class_logits = torch.stack([torch.stack(grouped_logits[label]).mean() for label in labels])
        class_scores = torch.softmax(class_logits, dim=0)
        score_by_label = {
            label: float(class_scores[index].item())
            for index, label in enumerate(labels)
        }

        top_k = min(ROUTE_TOP_K, len(labels))
        top_scores, top_indices = torch.topk(class_scores, k=top_k)
        raw_label = labels[int(top_indices[0])]
        raw_confidence = float(top_scores[0].item())
        final_label = raw_label if raw_confidence >= ROUTE_MIN_CONFIDENCE else "other"
        final_confidence = score_by_label[final_label]

        return RouteResult(
            label=final_label,
            raw_label=raw_label,
            confidence=final_confidence,
            raw_confidence=raw_confidence,
            scores=[
                ClassScore(label=labels[int(index)], score=float(score.item()))
                for score, index in zip(top_scores, top_indices)
            ],
        )


class EfficientNetDiseaseClassifier:
    def __init__(
        self,
        device: torch.device,
        *,
        model_id: str,
        model_subdir: str,
        model_filename: str,
        run_config_filename: str,
        model_revision: str | None,
        top_k: int,
        classifier_name: str,
    ) -> None:
        self.device = device
        self.classifier_name = classifier_name
        self.top_k = top_k
        self.model_path = _resolve_local_or_hf_file(
            model_id,
            filename=model_filename,
            subdir=model_subdir,
            revision=model_revision,
        )
        self.run_config_path = _resolve_local_or_hf_file(
            model_id,
            filename=run_config_filename,
            subdir=model_subdir,
            revision=model_revision,
        )

        with open(self.run_config_path, "r", encoding="utf-8") as fp:
            run_config = json.load(fp)

        self.class_names: List[str] = run_config.get("class_names") or []
        self.image_size = int(run_config.get("image_size", 224))
        if not self.class_names:
            raise RuntimeError(f"{self.classifier_name} run_config.json is missing class_names.")

        checkpoint = _torch_load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, len(self.class_names))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        mean = weights.transforms().mean
        std = weights.transforms().std
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(self.image_size * 1.14)),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        image_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(image_tensor)
        probs = torch.softmax(logits[0], dim=0)
        top_k = min(self.top_k, len(self.class_names))
        top_scores, top_indices = torch.topk(probs, k=top_k)
        predicted_label = self.class_names[int(top_indices[0])]
        confidence = float(top_scores[0].item())

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "is_normal": predicted_label == "정상",
            "scores": [
                ClassScore(label=self.class_names[int(index)], score=float(score.item()))
                for score, index in zip(top_scores, top_indices)
            ],
        }


class MultimodalAnalyzerService:
    def __init__(self) -> None:
        self.device = _resolve_device(DEVICE)
        logger.info("Loading multimodal models on device=%s", self.device)
        self.route_classifier = ImageRouteClassifier(self.device)
        self.eye_classifier = EfficientNetDiseaseClassifier(
            self.device,
            model_id=EYE_MODEL_ID,
            model_subdir=EYE_MODEL_SUBDIR,
            model_filename=EYE_MODEL_FILENAME,
            run_config_filename=EYE_RUN_CONFIG_FILENAME,
            model_revision=EYE_MODEL_REVISION,
            top_k=EYE_TOP_K,
            classifier_name="eye disease",
        )
        self.skin_classifier = self._load_optional_skin_classifier()
        logger.info(
            "Multimodal models loaded. route_model=%s eye_model=%s skin_model=%s",
            ROUTE_MODEL_ID,
            self.eye_classifier.model_path,
            self.skin_classifier.model_path if self.skin_classifier else "disabled",
        )

    def _load_optional_skin_classifier(self) -> EfficientNetDiseaseClassifier | None:
        if not SKIN_MODEL_ID:
            logger.info("Skin disease classifier disabled. SKIN_MODEL_ID is empty.")
            return None

        try:
            return EfficientNetDiseaseClassifier(
                self.device,
                model_id=SKIN_MODEL_ID,
                model_subdir=SKIN_MODEL_SUBDIR,
                model_filename=SKIN_MODEL_FILENAME,
                run_config_filename=SKIN_RUN_CONFIG_FILENAME,
                model_revision=SKIN_MODEL_REVISION,
                top_k=SKIN_TOP_K,
                classifier_name="skin disease",
            )
        except Exception as exc:
            logger.warning("Skin disease classifier is unavailable and will be disabled: %s", exc)
            return None

    def _download_image(self, image_url: str) -> Image.Image:
        response = requests.get(
            image_url,
            timeout=min(IMAGE_DOWNLOAD_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    def analyze(self, req: MultimodalAnalyzeRequest) -> MultimodalAnalyzeResponse:
        image = self._download_image(str(req.image_url))
        route_result = self.route_classifier.predict(image)
        eye_result: EyeDiseaseResult | None = None
        skin_result: SkinDiseaseResult | None = None
        if route_result.label == "eye_closeup":
            eye_result = EyeDiseaseResult(**self.eye_classifier.predict(image))
        elif route_result.label == "skin_closeup" and self.skin_classifier is not None:
            skin_result = SkinDiseaseResult(**self.skin_classifier.predict(image))

        return MultimodalAnalyzeResponse(
            image_url=str(req.image_url),
            analyze_at=datetime.datetime.now(SEOUL_TZ).isoformat(),
            route=route_result,
            eye_disease=eye_result,
            skin_disease=skin_result,
            processing={
                "device": str(self.device),
                "route_model_id": ROUTE_MODEL_ID,
                "route_model_pretrained": ROUTE_MODEL_PRETRAINED,
                "eye_model_id": EYE_MODEL_ID,
                "eye_model_filename": EYE_MODEL_FILENAME,
                "skin_model_id": SKIN_MODEL_ID or None,
                "skin_model_filename": SKIN_MODEL_FILENAME if SKIN_MODEL_ID else None,
                "skin_classifier_enabled": self.skin_classifier is not None,
            },
        )
