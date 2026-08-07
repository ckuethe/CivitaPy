from enum import Enum, auto


class ModelType(str, Enum):
    CHECKPOINT = "Checkpoint"
    TEXTUAL_INVERSION = "TextualInversion"
    HYPERNETWORK = "Hypernetwork"
    AESTHETIC_GRADIENT = "AestheticGradient"
    LORA = "LORA"
    LOCON = "LoCon"
    DORA = "DoRA"
    CONTROLNET = "Controlnet"
    UPSCALER = "Upscaler"
    MOTION_MODULE = "MotionModule"
    VAE = "VAE"
    POSES = "Poses"
    WILDCARDS = "Wildcards"
    WORKFLOWS = "Workflows"
    DETECTION = "Detection"
    OTHER = "Other"


class ModelFileType(str, Enum):
    MODEL = "Model"
    TEXT_ENCODER = "Text Encoder"
    PRUNED_MODEL = "Pruned Model"
    NEGATIVE = "Negative"
    TRAINING_DATA = "Training Data"
    VAE = "VAE"
    CONFIG = "Config"
    ARCHIVE = "Archive"


class BaseModel(str, Enum):
    SD_1_5 = "SD 1.5"
    SD_2_1 = "SD 2.1"
    SD_3_5 = "SD 3.5"
    SDXL_1_0 = "SDXL 1.0"
    FLUX_1_D = "Flux.1 D"
    ILLUSTRIOUS = "Illustrious"
    PONY = "Pony"
    HUNYUAN_VIDEO = "Hunyuan Video"


class BaseModelType(str, Enum):
    STANDARD = "Standard"
    INPAINTING = "Inpainting"
    REFINER = "Refiner"
    PIX2PIX = "Pix2Pix"


class SortOrder(str, Enum):
    HIGHEST_RATED = "Highest Rated"
    MOST_DOWNLOADED = "Most Downloaded"
    NEWEST = "Newest"
    OLDEST = "Oldest"
    MOST_LIKED = "Most Liked"
    MOST_COMMENTED = "Most Commented"


class ImageSortOrder(str, Enum):
    MOST_REACTIONS = "Most Reactions"
    MOST_COMMENTS = "Most Comments"
    MOST_COLLECTED = "Most Collected"
    NEWEST = "Newest"
    OLDEST = "Oldest"
    RANDOM = "Random"


class ArticleSortOrder(str, Enum):
    NEWEST = "Newest"
    RECENTLY_UPDATED = "Recently Updated"
    MOST_REACTIONS = "Most Reactions"
    MOST_COMMENTS = "Most Comments"
    MOST_BOOKMARKS = "Most Bookmarks"
    MOST_COLLECTED = "Most Collected"


class CollectionSortOrder(str, Enum):
    NEWEST = "Newest"
    MOST_FOLLOWERS = "Most Followers"


class VaultSortOrder(str, Enum):
    RECENTLY_ADDED = "Recently Added"
    RECENTLY_CREATED = "Recently Created"
    MODEL_NAME = "Model Name"
    MODEL_SIZE = "Model Size"


class Period(str, Enum):
    ALL_TIME = "AllTime"
    YEAR = "Year"
    MONTH = "Month"
    WEEK = "Week"
    DAY = "Day"


class CheckpointType(str, Enum):
    STANDARD = "Standard"
    TRAINED = "Trained"
    MERGE = "Merge"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class NSFWLevel(str, Enum):
    NONE = "None"
    SOFT = "Soft"
    MATURE = "Mature"
    X = "X"


class UserTier(str, Enum):
    FREE = "free"
    FOUNDER = "founder"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class UserStatus(str, Enum):
    ACTIVE = "active"
    MUTED = "muted"
    BANNED = "banned"


class ModelVersionStatus(str, Enum):
    PUBLISHED = "Published"
    DRAFT = "Draft"
    UNPUBLISHED = "Unpublished"


class Availability(str, Enum):
    PUBLIC = "Public"
    PRIVATE = "Private"
    SCHEDULED = "Scheduled"


class ModerationMode(str, Enum):
    ARCHIVED = "Archived"
    TAKEN_DOWN = "TakenDown"


class VaultStatus(str, Enum):
    PENDING = "Pending"
    STORED = "Stored"
    FAILED = "Failed"


class AvatarNSFWLevel(str, Enum):
    NONE = "None"
    SOFT = "Soft"
    MATURE = "Mature"
    X = "X"
