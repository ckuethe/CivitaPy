from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

T = TypeVar("T")


def _as_int(value: Any) -> Any:
    """Coerce bool/int/float/numeric-string to ``int`` for tolerant parsing.

    Civitai occasionally reports a count as a float (e.g. ``156742.0``) or ``null``
    (``None``) instead of an int, which would otherwise raise a ``ValidationError``.
    ``None`` becomes ``0`` (the field default); anything else is coerced to its
    integer value. Values that can't be sensibly converted are treated as ``0`` and
    a warning is logged — these stats are informational and not worth failing a
    parse over.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        try:
            return int(float(s))
        except ValueError:
            logger.warning("Could not coerce stat value %r to an integer; defaulting to 0", value)
            return 0
    logger.warning("Could not coerce stat value %r to an integer; defaulting to 0", value)
    return 0


class PaginationMetadata(BaseModel):
    """Pagination metadata returned in list endpoint responses.

    Contains cursor-based and page-based pagination info depending on the request.
    Not every endpoint reports totalItems/totalPages — some report 0 when an exact count is expensive.
    Always prefer nextCursor or nextPage to drive "load more" UIs over relying on counts.
    """

    next_cursor: str | None = Field(default=None, alias="nextCursor")
    next_page: str | None = Field(default=None, alias="nextPage")
    current_page: int | None = Field(default=None, alias="currentPage")
    page_size: int | None = Field(default=None, alias="pageSize")
    total_items: int | None = Field(default=None, alias="totalItems")
    total_pages: int | None = Field(default=None, alias="totalPages")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic envelope for paginated list responses.

    Contains a list of items and their associated pagination metadata.
    """

    items: list[T]
    metadata: PaginationMetadata


class Stats(BaseModel):
    """Aggregate statistics for models (download counts, ratings, tips)."""

    download_count: int = Field(alias="downloadCount")
    thumbs_up_count: int = Field(default=0, alias="thumbsUpCount")
    thumbs_down_count: int = Field(default=0, alias="thumbsDownCount")
    comment_count: int = Field(default=0, alias="commentCount")
    tipped_amount_count: int = Field(default=0, alias="tippedAmountCount")

    @field_validator(
        "download_count",
        "thumbs_up_count",
        "thumbs_down_count",
        "comment_count",
        "tipped_amount_count",
        mode="before",
    )
    @classmethod
    def _coerce_count(cls, value: Any) -> Any:
        return _as_int(value)


class UserSummary(BaseModel):
    """Minimal user information returned in creator/author fields across resources."""

    id: int | None = None
    username: str | None = None
    image: str | None = None


class ModelFileHashes(BaseModel):
    """Cryptographic hashes for a model file.

    Hash values are uppercase strings. SHA256 is always 64-char hex; AutoV1/AutoV2/etc may be shorter digests.
    """

    auto_v1: str | None = Field(default=None, alias="AutoV1")
    auto_v2: str | None = Field(default=None, alias="AutoV2")
    sha256: str | None = Field(default=None, alias="SHA256")
    crc32: str | None = Field(default=None, alias="CRC32")
    blake3: str | None = Field(default=None, alias="BLAKE3")
    auto_v3: str | None = Field(default=None, alias="AutoV3")


class ModelFileMetadata(BaseModel):
    """Additional metadata about a model file (format, size class, precision)."""

    format: str | None = None  # e.g. "SafeTensor"
    size: str | None = None  # e.g. "pruned"
    fp: str | None = None  # e.g. "fp16", "bf16"


class ModelVersionFile(BaseModel):
    """A single file within a model version (checkpoint, LoRA weight, etc.).

    Includes download URL, hashes for integrity verification, and scan results.
    The primary flag indicates the main downloadable file.
    """

    id: int
    name: str
    type: str  # ModelFileType enum value (e.g. "Model", "VAE", "LORA")
    size_kb: float = Field(alias="sizeKB")
    metadata: ModelFileMetadata | None = None
    pickle_scan_result: str | None = Field(default=None, alias="pickleScanResult")  # e.g. "Success"
    virus_scan_result: str | None = Field(default=None, alias="virusScanResult")  # e.g. "Success"
    hashes: dict[str, str] | None = None
    download_url: str | None = Field(default=None, alias="downloadUrl")  # /api/download/models/{id}
    primary: bool = False


class VersionStats(BaseModel):
    """Download and rating statistics for a model version."""

    download_count: int = Field(alias="downloadCount")
    thumbs_up_count: int = Field(default=0, alias="thumbsUpCount")

    @field_validator("download_count", "thumbs_up_count", mode="before")
    @classmethod
    def _coerce_count(cls, value: Any) -> Any:
        return _as_int(value)


# ---------------------------------------------------------------------------
# Model (top-level) models
# ---------------------------------------------------------------------------


class ModelVersionSummary(BaseModel):
    """Condensed representation of a model version as returned in GET /models list responses.

    Contains the essential identity and stats but may omit full file lists depending on primaryFileOnly.
    """

    id: int
    name: str
    base_model: str = Field(alias="baseModel")  # e.g. "Illustrious", "SDXL 1.0"
    base_model_type: str | None = Field(default=None, alias="baseModelType")  # Standard, Inpainting, Refiner, Pix2Pix
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    supports_generation: bool = False
    stats: VersionStats | None = None
    files: list[ModelVersionFile] = []


class Model(BaseModel):
    """A CivitAI model (checkpoint, LoRA, VAE, embedding, controlnet, etc.).

    Returned by GET /models and GET /models/{id}. Contains a summary of all versions.
    The mode field indicates moderation state: "Archived" drops files/downloadUrl; "TakenDown" also drops images.
    """

    id: int
    name: str
    description: str = ""  # HTML string
    type: str  # ModelType enum value (e.g. "Checkpoint", "LORA", "VAE")
    nsfw: bool = False
    nsfw_level: int = Field(default=0, alias="nsfwLevel")
    availability: str | None = None  # Public, Private, Scheduled
    supports_generation: bool = False  # Supported by on-site generation workflows
    allow_no_credit: bool = True
    allow_commercial_use: str | None = None  # e.g. "{Image,RentCivit}" or empty string for "No"
    allow_derivatives: bool = True
    allow_different_license: bool = True
    minor: bool = False
    poi: bool = False  # Person of Interest flag
    sfw_only: bool = False
    mode: str | None = None  # Moderation state: "Archived", "TakenDown" (null if healthy)
    stats: Stats | None = None
    creator: UserSummary | None = None
    tags: list[str] = []
    model_versions: list[ModelVersionSummary] = Field(default_factory=list, alias="modelVersions")


# ---------------------------------------------------------------------------
# Model version (full detail)
# ---------------------------------------------------------------------------


class FullModelFile(ModelVersionFile):
    """Extended file representation used in the mini endpoint responses.

    Inherits all fields from ModelVersionFile plus an additional hash_value field.
    """

    hash_value: str | None = Field(default=None)  # Primary file hash


class MiniHashes(BaseModel):
    """Cryptographic hashes as returned by the GET /model-versions/mini endpoint."""

    auto_v1: str | None = Field(default=None, alias="AutoV1")
    auto_v2: str | None = Field(default=None, alias="AutoV2")
    sha256: str | None = Field(default=None, alias="SHA256")
    crc32: str | None = Field(default=None, alias="CRC32")
    blake3: str | None = Field(default=None, alias="BLAKE3")
    auto_v3: str | None = Field(default=None, alias="AutoV3")


class MiniModelFile(BaseModel):
    """Minimal file metadata as returned in vault item listings."""

    id: int
    size_kb: float = Field(alias="sizeKB")
    url: str  # Download URL for stored items
    display_name: str = Field(alias="displayName")


class ModelVersion(BaseModel):
    """Full model version detail (checkpoint release, LoRA weight set, etc.).

    Returned by GET /model-versions/{id}. Contains complete file list with hashes,
    trained words, AIR identifier, and associated images. The air field is the canonical
    URN-style reference for this version across Civitai APIs.
    """

    id: int
    model_id: int = Field(alias="modelId")
    name: str  # e.g. "v16.0"
    description: str | None = None
    base_model: str = Field(alias="baseModel")  # e.g. "Illustrious", "SDXL 1.0"
    base_model_type: str | None = Field(default=None, alias="baseModelType")  # Standard, Inpainting, Refiner, Pix2Pix
    air: str | None = None  # Canonical AIR URN (e.g. urn:air:sdxl:checkpoint:civitai:827184@2514310)
    status: str | None = None  # Published, Draft, Unpublished
    availability: str | None = None  # Public, Private, Scheduled
    nsfw_level: int = Field(default=0, alias="nsfwLevel")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    upload_type: str | None = Field(default=None, alias="uploadType")  # Created, Trained, etc.
    usage_control: str | None = Field(default=None, alias="usageControl")  # e.g. "Download"
    trained_words: list[str] = []  # Words that influence the model output
    early_access_config: Any | None = None
    early_access_ends_at: datetime | None = Field(default=None, alias="earlyAccessEndsAt")
    training_status: str | None = Field(default=None, alias="trainingStatus")
    training_details: Any | None = None
    stats: VersionStats | None = None  # Only downloadCount and thumbsUpCount for versions
    model: dict[str, Any] | None = None  # Parent model summary (name, type, nsfw)
    files: list[ModelVersionFile] = []  # All downloadable files with hashes
    images: list[dict[str, Any]] = []  # Associated preview/thumbnail images
    download_url: str | None = Field(default=None, alias="downloadUrl")


class MiniModelVersion(BaseModel):
    """Minimal model version info as returned by GET /model-versions/mini/{id}.

    A trimmed representation useful for quick lookups without full file/image lists.
    The canGenerate field indicates if this resource can be used in Orchestration workflows.
    checkPermission is true when the resource is gated (early-access or Private).
    """

    air: str | None = None  # Canonical AIR URN
    version_name: str = Field(alias="versionName")
    model_name: str = Field(alias="modelName")
    base_model: str = Field(alias="baseModel")
    availability: str | None = None
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    size: float | None = None  # File size in KB
    file_type: str | None = Field(default=None, alias="fileType")  # e.g. "Model", "VAE"
    file_name: str | None = Field(default=None, alias="fileName")
    hashes: MiniHashes | None = None
    download_urls: list[str] = Field(alias="downloadUrls")
    format: str | None = None  # e.g. "SafeTensor", "ckpt"
    can_generate: bool = False  # Usable in Orchestration workflows
    is_featured: bool = False
    require_auth: bool = False  # downloadUrls need a bearer token when true
    check_permission: bool = False  # Resource is gated (early-access or Private)
    early_access_ends_at: datetime | None = Field(default=None, alias="earlyAccessEndsAt")
    free_trial_limit: int | None = None  # Number of generations before requiring payment
    additional_resource_charge: bool = False
    minor: bool = False
    sfw_only: bool = False


# ---------------------------------------------------------------------------
# Hash lookup responses
# ---------------------------------------------------------------------------


class HashLookupResult(BaseModel):
    """Mapping result from POST /model-versions/by-hash/ids.

    Returns only the model version ID and hash for each matched input, without full file details.
    Unmatched hashes are silently dropped — response can have fewer entries than request.
    """

    model_version_id: int = Field(alias="modelVersionId")
    hash: str


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


class ImageStats(BaseModel):
    """Engagement statistics for an image (reactions, likes, hearts)."""

    cry_count: int = 0
    laugh_count: int = 0
    like_count: int = 0
    dislike_count: int = 0
    heart_count: int = 0
    comment_count: int = 0


class CivitaiResourceRef(BaseModel):
    """Reference to a CivitAI model version embedded in an image's generation metadata.

    Links back to the specific model (checkpoint, LoRA) used to generate this image.
    The weight field is only present for LoRAs and indicates how heavily it was weighted.
    """

    type: str  # e.g. "checkpoint", "lora"
    model_version_id: int | None = Field(default=None, alias="modelVersionId")
    weight: float | None = None


class ImageMeta(BaseModel):
    """Generation metadata embedded in an image (prompt, sampler settings, resources used).

    Only present when the uploader included metadata at post time. The civitaiResources field
    maps each referenced resource to its CivitAI modelVersionId for cross-referencing.
    Top-level modelVersionIds is a deduplicated list of every version in civitaiResources.
    """

    size: str | None = None  # e.g. "832x1216"
    seed: int | None = None
    steps: int | None = None
    sampler: str | None = None  # e.g. "DPM++ 2M", "Euler a"
    cfg_scale: float | None = Field(default=None, alias="cfgScale")
    clip_skip: int | None = Field(default=None, alias="clipSkip")
    prompt: str | None = None
    negative_prompt: str | None = Field(default=None, alias="negativePrompt")
    resources: list[dict[str, Any]] = []  # External resource references (opaque)
    civitai_resources: list[CivitaiResourceRef] = Field(default_factory=list, alias="civitaiResources")


class Image(BaseModel):
    """An image generated and shared on CivitAI.

    Returned by GET /images. The hash field is a BlurHash for placeholder rendering.
    meta contains generation settings (prompt, sampler) when included by the uploader.
    modelVersionIds is a deduplicated list of all CivitAI versions referenced in the image's metadata.
    """

    id: int
    url: str  # CDN URL to the full-size image
    hash: str  # BlurHash for placeholder rendering (not cryptographic)
    width: int | None = None
    height: int | None = None
    type: str = "image"  # "image", "video", "audio"
    nsfw: bool = False
    nsfw_level: str | None = Field(default=None, alias="nsfwLevel")  # "None", "Soft", "Mature", "X"
    browsing_level: int | None = Field(default=None, alias="browsingLevel")  # Raw integer bitmask
    created_at: datetime = Field(alias="createdAt")
    post_id: int | None = Field(default=None, alias="postId")
    username: str | None = None  # Uploader's username
    base_model: str | None = Field(default=None, alias="baseModel")  # e.g. "SDXL 1.0"
    model_version_ids: list[int] = Field(
        default_factory=list, alias="modelVersionIds"
    )  # All referenced CivitAI version IDs (deduplicated)
    stats: ImageStats | None = None
    meta: ImageMeta | None = None


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


class ArticleTag(BaseModel):
    """A tag associated with an article."""

    id: int
    name: str
    is_category: bool = Field(default=False, alias="isCategory")


class CoverImage(BaseModel):
    """Cover image for an article listing.

    Contains CDN URL and dimensions for display in list views.
    """

    id: int
    url: str  # CDN URL to the cover image
    nsfw_level: int | None = Field(default=None, alias="nsfwLevel")
    width: int | None = None
    height: int | None = None


class ArticleStats(BaseModel):
    """Engagement statistics for an article."""

    favorite_count: int = 0
    collected_count: int = 0
    comment_count: int = 0
    like_count: int = 0
    heart_count: int = 0
    view_count: int = 0
    tipped_amount_count: int = Field(default=0, alias="tippedAmountCount")

    @field_validator(
        "favorite_count",
        "collected_count",
        "comment_count",
        "like_count",
        "heart_count",
        "view_count",
        "tipped_amount_count",
        mode="before",
    )
    @classmethod
    def _coerce_count(cls, value: Any) -> Any:
        return _as_int(value)


class Article(BaseModel):
    """A tutorial or guide published on CivitAI.

    Returned by GET /articles and GET /articles/{id}. The list endpoint returns a summary;
    the detail endpoint additionally includes content (HTML body). Only published articles are returned.
    """

    id: int
    title: str
    published_at: datetime | None = Field(default=None, alias="publishedAt")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    nsfw_level: int = 0
    availability: str | None = None
    status: str | None = None  # Published, Draft
    stats: ArticleStats | None = None
    user: UserSummary | None = None  # Author information
    tags: list[ArticleTag] = []  # Tags with category flag
    cover_image: CoverImage | None = Field(default=None, alias="coverImage")
    content: str | None = None  # HTML body (only in detail endpoint)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class Collection(BaseModel):
    """A user-curated collection of models or other resources on CivitAI.

    Returned by GET /collections and GET /collections/{id}. Only public collections are returned;
    private ones appear as missing (404). The type field indicates what kind of items the collection holds.
    itemCount counts only accepted/added items in the collection.
    """

    id: int
    name: str
    description: str | None = None
    type: str  # "Model" or other resource types
    nsfw_level: int = 0
    read: str | None = None  # Access level (e.g. "Public")
    is_public: bool = Field(default=False, alias="isPublic")
    item_count: int = 0  # Number of accepted items in the collection
    cover_image_url: str | None = Field(default=None, alias="coverImageUrl")  # CDN URL or null for mature clamped out
    user: UserSummary | None = None  # Collection author
    tags: list[ArticleTag] = []


# ---------------------------------------------------------------------------
# Creators
# ---------------------------------------------------------------------------


class Creator(BaseModel):
    """A model creator/author on CivitAI.

    Returned by GET /creators. Only creators with published models are included;
    creators with no published models are excluded entirely from the listing.
    Sorted alphabetically by username — use query= to scope results rather than linear paging.
    """

    username: str  # Auto-slugified, used in URL paths and filter params
    model_count: int | None = Field(default=None, alias="modelCount")  # Only included when > 0
    link: str | None = None  # API link to this creator's models (GET /models?username=...)
    image: str | None = None  # Avatar CDN URL or null if no avatar set


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TagItem(BaseModel):
    """A model tag returned by GET /tags.

    Only includes tags used for models (entityType=Model). Total counts may be reported as 0
    when exact counting is expensive — always use nextPage to drive pagination instead of totalItems.
    """

    name: str
    link: str | None = None  # API link to filter models by this tag


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------


class UserLite(BaseModel):
    """Minimal user lookup result from GET /users.

    Returns only {id, username} per result — intentionally lean endpoint for batch lookups.
    Deleted and system users (id=-1) are filtered out automatically.
    """

    id: int
    username: str
    avatar_nsfw: str = Field(default="None", alias="avatarNsfw")  # "None", "Soft", "Mature", "X"


class CurrentUser(BaseModel):
    """Authenticated user profile returned by GET /me.

    Requires a valid API token via Authorization header. Returns the caller's account info,
    subscription status, and tier information. Uses extra="allow" to accept additional fields
    that may be present in responses but aren't documented (e.g., email, buzzLimit).
    """

    model_config = {"extra": "allow"}

    id: int
    username: str
    status: str | None = None  # active, muted, banned
    is_member: bool = Field(default=False, alias="isMember")  # True when tier != 'free'
    subscriptions: list[str] = []


# ---------------------------------------------------------------------------
# Permissions & Vault
# ---------------------------------------------------------------------------


class VaultInfo(BaseModel):
    """Vault storage information returned by GET /vault/get.

    Only available to paid members (bronze, silver, gold, founder tiers). Free-tier callers get null.
    usedStorageKb is the sum of modelSizeKb + detailsSizeKb + imagesSizeKb across all vault items.
    """

    user_id: int = Field(alias="userId")  # Owner's ID and vault primary key
    storage_kb: float = Field(alias="storageKb")  # Total allowance from active membership(s)
    used_storage_kb: float = Field(alias="usedStorageKb")  # Sum of all item sizes in the vault
    meta: dict[str, Any] | None = None
    updated_at: datetime = Field(alias="updatedAt")


class VaultItem(BaseModel):
    """An item stored in a user's CivitAI vault.

    Returned by GET /vault/all and GET /vault/check-vault (when present). Status indicates ingestion state:
    Pending → Stored → Failed (if processing fails). Cover image and full files are only available once Stored.
    modelName, versionName, creatorName are snapshots at vault time — they survive original model deletion.
    """

    id: int
    vault_id: int = Field(alias="vaultId")
    status: str  # VaultStatus enum value (Pending, Stored, Failed)
    model_version_id: int | None = Field(default=None, alias="modelVersionId")
    model_id: int | None = Field(default=None, alias="modelId")
    model_name: str = Field(alias="modelName")  # Snapshot at vault time (survives deletion)
    version_name: str = Field(alias="versionName")  # Snapshot at vault time
    creator_id: int | None = Field(default=None, alias="creatorId")
    creator_name: str | None = Field(default=None, alias="creatorName")  # Snapshot at vault time
    type: str | None = None  # ModelType enum value (e.g. "Checkpoint", "LORA")
    base_model: str | None = Field(default=None, alias="baseModel")  # e.g. "SDXL 1.0"
    category: str | None = None  # Category tag (if any)
    model_size_kb: float = Field(alias="modelSizeKb")
    details_size_kb: float = Field(default=0.0, alias="detailsSizeKb")
    images_size_kb: float = Field(default=0.0, alias="imagesSizeKb")
    created_at: datetime | None = Field(default=None, alias="createdAt")  # Model version creation time
    added_at: datetime | None = Field(default=None, alias="addedAt")  # When item was vaulted
    refreshed_at: datetime | None = Field(default=None, alias="refreshedAt")  # Last content refresh (null if not done)
    notes: str | None = None  # User-provided notes for this vault entry
    meta: dict[str, Any] | None = None  # Includes failures counter for ingestion retries
    cover_image_url: str | None = Field(default=None, alias="coverImageUrl")  # CDN URL (only when Stored)
    files: list[MiniModelFile] = []  # File metadata with download URLs (only when Stored)


class VaultToggleResponse(BaseModel):
    """Response from POST /vault/toggle-version.

    Idempotent operation — adding an already-present item or removing a missing one both succeed silently.
    The vaultId field is omitted when the operation removed the item (since there's no longer a vault).
    """

    success: bool
    vault_id: int | None = Field(default=None, alias="vaultId")  # Omitted on removal
