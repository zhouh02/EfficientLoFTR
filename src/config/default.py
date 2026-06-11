from yacs.config import CfgNode as CN
_CN = CN()

##############  ↓  LoFTR Pipeline  ↓  ##############
_CN.LOFTR = CN()
_CN.LOFTR.BACKBONE_TYPE = 'RepVGG'
_CN.LOFTR.ALIGN_CORNER = False
_CN.LOFTR.RESOLUTION = (8, 1)
_CN.LOFTR.FINE_WINDOW_SIZE = 8  # window_size in fine_level, must be even
_CN.LOFTR.MP = False
_CN.LOFTR.REPLACE_NAN = False
_CN.LOFTR.EVAL_TIMES = 1
_CN.LOFTR.HALF = False

# 1. LoFTR-backbone (local feature CNN) config
_CN.LOFTR.BACKBONE = CN()
_CN.LOFTR.BACKBONE.BLOCK_DIMS = [64, 128, 256]  # s1, s2, s3

# 2. LoFTR-coarse module config
_CN.LOFTR.COARSE = CN()
_CN.LOFTR.COARSE.D_MODEL = 256
_CN.LOFTR.COARSE.D_FFN = 256
_CN.LOFTR.COARSE.NHEAD = 8
_CN.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.LOFTR.COARSE.AGG_SIZE0 = 4
_CN.LOFTR.COARSE.AGG_SIZE1 = 4
_CN.LOFTR.COARSE.NO_FLASH = False
_CN.LOFTR.COARSE.ROPE = True
_CN.LOFTR.COARSE.NPE = None # [832, 832, long_side, long_side] Suggest setting based on the long side of the input image, especially when the long_side > 832

# 3. Coarse-Matching config
_CN.LOFTR.MATCH_COARSE = CN()
_CN.LOFTR.MATCH_COARSE.THR = 0.2 # recommend 0.2 for full model and 25 for optimized model
_CN.LOFTR.MATCH_COARSE.BORDER_RM = 2
_CN.LOFTR.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.2  # training tricks: save GPU memory
_CN.LOFTR.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock
_CN.LOFTR.MATCH_COARSE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_COARSE.SKIP_SOFTMAX = False
_CN.LOFTR.MATCH_COARSE.FP16MATMUL = False

_CN.LOFTR.FINE = CN()
_CN.LOFTR.FINE.D_MODEL = 64 #56
_CN.LOFTR.FINE.D_FFN = 64 #56
_CN.LOFTR.FINE.NHEAD = 4
_CN.LOFTR.FINE.LAYER_NAMES = ['cross'] * 2

# 4. Fine-Matching config
_CN.LOFTR.MATCH_FINE = CN()
_CN.LOFTR.MATCH_FINE.SPARSE_SPVS = True
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_TEMPERATURE = 10.0  # ✅ 修改：1.0 -> 10.0 (CoMatch要求)
_CN.LOFTR.MATCH_FINE.LOCAL_REGRESS_SLICEDIM = 64  # ✅ 修改：8 -> 64 (高精度，如果显存不足可改回8)

# 边界约束配置（保留你的原始配置）
_CN.LOFTR.MATCH_FINE.MAX_OFFSET_RATIO = 1.5
_CN.LOFTR.MATCH_FINE.ENABLE_BOUNDARY_CHECK = True
_CN.LOFTR.MATCH_FINE.CONSISTENCY_THRESHOLD = 2.0

# 5. LoFTR Losses
# -- # coarse-level
_CN.LOFTR.LOSS = CN()
_CN.LOFTR.LOSS.COARSE_TYPE = 'focal'  # ['focal', 'cross_entropy']
_CN.LOFTR.LOSS.COARSE_WEIGHT = 1.0
_CN.LOFTR.LOSS.COARSE_SIGMOID_WEIGHT = 1.0
_CN.LOFTR.LOSS.COARSE_OVERLAP_WEIGHT = True  # ✅ 修改：False -> True (CoMatch推荐)
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT = True  # ✅ 修改：False -> True (CoMatch推荐)
_CN.LOFTR.LOSS.FINE_OVERLAP_WEIGHT2 = False

# -- - -- # focal loss (coarse)
_CN.LOFTR.LOSS.FOCAL_ALPHA = 0.25
_CN.LOFTR.LOSS.FOCAL_GAMMA = 2.0
_CN.LOFTR.LOSS.POS_WEIGHT = 1.0
_CN.LOFTR.LOSS.NEG_WEIGHT = 1.0

# -- # fine-level (第一阶段)
_CN.LOFTR.LOSS.FINE_TYPE = 'l2'  # ✅ 修改：'l2_with_std' -> 'l2' (CoMatch使用l2)
_CN.LOFTR.LOSS.FINE_WEIGHT = 1.0
_CN.LOFTR.LOSS.FINE_CORRECT_THR = 1.0  # for filtering valid fine-level gts (some gt matches might fall out of the fine-level window)

# -- # subpixel-level (第二阶段 - 原始单边损失权重，作为备用)
_CN.LOFTR.LOSS.LOCAL_WEIGHT = 0.5

# ============== 新增：第二阶段双边细化损失配置 ==============
# -- # bilateral refinement loss (双边偏移损失 - 主要监督)
_CN.LOFTR.LOSS.BILATERAL_WEIGHT = 1.0
"""
双边偏移损失权重配置说明：
- 1.0: 标准配置，与其他阶段损失平衡（推荐）
- 1.5-2.0: 如果第二阶段精度不够，可以加大
- 0.5: 如果过拟合第二阶段，可以减小
- 0.0: 禁用双边损失，回退到原始单边监督
"""

# -- # similarity contrastive loss (相似度对比损失 - 辅助监督)
_CN.LOFTR.LOSS.SIMILARITY_WEIGHT = 0.0  # ✅ 修改：0.1 -> 0.0 (建议初期不使用，稳定后再开启)
"""
相似度对比损失权重配置说明：
- 0.0: 不使用相似度监督（只用双边偏移损失）
- 0.05-0.1: 轻度辅助监督（推荐初学者）
- 0.15: 中度辅助监督（推荐标准配置）
- 0.2: 强辅助监督（追求极致精度）
- >0.3: 不推荐，可能导致过度关注相似度而忽略偏移准确性
"""

# -- # 第二阶段相关配置
_CN.LOFTR.LOSS.STAGE2_CORRECT_THR = 2.0
"""
第二阶段正确匹配阈值（像素）：
用于判断第二阶段预测是否为正确匹配，影响相似度对比损失的正负样本划分
- 1.0: 严格标准（高精度要求）
- 2.0: 标准配置（推荐）
- 3.0: 宽松标准（增加正样本数量）
"""

# -- # 训练策略配置
_CN.LOFTR.LOSS.BILATERAL_WARMUP_STEPS = 0
"""
双边损失warmup步数：
- 0: 从一开始就使用双边损失（推荐）
- 5000-10000: 前N步先训练其他阶段，再加入双边监督
"""

_CN.LOFTR.LOSS.SIMILARITY_WARMUP_STEPS = 0
"""
相似度损失warmup步数：
- 0: 从一开始就使用相似度损失
- 10000-20000: 前N步先训练偏移，再加入相似度监督（推荐策略）
"""

# -- # 监控和调试配置
_CN.LOFTR.LOSS.LOG_BILATERAL_STATS = True
"""
是否记录双边损失的详细统计信息：
- True: 记录 loss_l_left, loss_l_right, avg_similarity 等
- False: 只记录汇总的 loss_bilateral
"""

_CN.LOFTR.LOSS.LOG_SIMILARITY_DISTRIBUTION = True
"""
是否记录相似度分布统计：
- True: 记录 avg_pos_similarity, avg_neg_similarity, n_pos/neg_samples
- False: 只记录 avg_similarity
"""
# ============== 双边细化损失配置结束 ==============


##############  Dataset  ##############
_CN.DATASET = CN()
# 1. data config
# training and validating
_CN.DATASET.TRAINVAL_DATA_SOURCE = None  # options: ['ScanNet', 'MegaDepth']
_CN.DATASET.TRAIN_DATA_ROOT = None
_CN.DATASET.TRAIN_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TRAIN_NPZ_ROOT = None
_CN.DATASET.TRAIN_LIST_PATH = None
_CN.DATASET.TRAIN_INTRINSIC_PATH = None
_CN.DATASET.VAL_DATA_ROOT = None
_CN.DATASET.VAL_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.VAL_NPZ_ROOT = None
_CN.DATASET.VAL_LIST_PATH = None    # None if val data from all scenes are bundled into a single npz file
_CN.DATASET.VAL_INTRINSIC_PATH = None
_CN.DATASET.FP16 = False
# testing
_CN.DATASET.TEST_DATA_SOURCE = None
_CN.DATASET.TEST_DATA_ROOT = None
_CN.DATASET.TEST_POSE_ROOT = None  # (optional directory for poses)
_CN.DATASET.TEST_NPZ_ROOT = None
_CN.DATASET.TEST_LIST_PATH = None   # None if test data from all scenes are bundled into a single npz file
_CN.DATASET.TEST_INTRINSIC_PATH = None

# 2. dataset config
# general options
_CN.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.4  # discard data with overlap_score < min_overlap_score
_CN.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
_CN.DATASET.AUGMENTATION_TYPE = None  # options: [None, 'dark', 'mobile']

# scanNet options
_CN.DATASET.SCAN_IMG_RESIZEX = 832  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.SCAN_IMG_RESIZEY = 480  # resize the shorter side, zero-pad bottom-right to square.

# MegaDepth options
_CN.DATASET.MGDPT_IMG_RESIZE = 832  # resize the longer side, zero-pad bottom-right to square.
_CN.DATASET.MGDPT_IMG_PAD = True  # pad img to square with size = MGDPT_IMG_RESIZE
_CN.DATASET.MGDPT_DEPTH_PAD = True  # pad depthmap to square with size = 2000
_CN.DATASET.MGDPT_DF = 8

_CN.DATASET.NPE_NAME = None

##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 64
_CN.TRAINER.CANONICAL_LR = 8e-3  # ✅ 修改：6e-3 -> 8e-3 (CoMatch训练策略)
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.FIND_LR = False  # use learning rate finder from pytorch-lightning

# optimizer
_CN.TRAINER.OPTIMIZER = "adamw"  # [adam, adamw]
_CN.TRAINER.TRUE_LR = None  # this will be calculated automatically at runtime
_CN.TRAINER.ADAM_DECAY = 0.  # ADAM: for adam
_CN.TRAINER.ADAMW_DECAY = 0.1

# step-based warm-up
_CN.TRAINER.WARMUP_TYPE = 'linear'  # [linear, constant]
_CN.TRAINER.WARMUP_RATIO = 0.1  # ✅ 修改：0. -> 0.1 (CoMatch训练策略)
_CN.TRAINER.WARMUP_STEP = 1875  # ✅ 修改：4800 -> 1875 (3 epochs, CoMatch训练策略)

# learning rate scheduler
_CN.TRAINER.SCHEDULER = 'MultiStepLR'  # [MultiStepLR, CosineAnnealing, ExponentialLR]
_CN.TRAINER.SCHEDULER_INTERVAL = 'epoch'    # [epoch, step]
_CN.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]  # ✅ 修改：[3, 6, 9, 12] -> [8, 12, 16, 20, 24] (CoMatch训练策略)
_CN.TRAINER.MSLR_GAMMA = 0.5
_CN.TRAINER.COSA_TMAX = 30  # COSA: CosineAnnealing
_CN.TRAINER.ELR_GAMMA = 0.999992  # ELR: ExponentialLR, this value for 'step' interval

# plotting related
_CN.TRAINER.ENABLE_PLOTTING = True
_CN.TRAINER.N_VAL_PAIRS_TO_PLOT = 32     # number of val/test paris for plotting
_CN.TRAINER.PLOT_MODE = 'evaluation'  # ['evaluation', 'confidence']
_CN.TRAINER.PLOT_MATCHES_ALPHA = 'dynamic'

# geometric metrics and pose solver
_CN.TRAINER.EPI_ERR_THR = 5e-4  # recommendation: 5e-4 for ScanNet, 1e-4 for MegaDepth (from SuperGlue)
_CN.TRAINER.POSE_GEO_MODEL = 'E'  # ['E', 'F', 'H']
_CN.TRAINER.POSE_ESTIMATION_METHOD = 'RANSAC'  # [RANSAC, LO-RANSAC]
_CN.TRAINER.RANSAC_PIXEL_THR = 0.5
_CN.TRAINER.RANSAC_CONF = 0.99999
_CN.TRAINER.RANSAC_MAX_ITERS = 10000
_CN.TRAINER.USE_MAGSACPP = False

# data sampler for train_dataloader
_CN.TRAINER.DATA_SAMPLER = 'scene_balance'  # options: ['scene_balance', 'random', 'normal']
# 'scene_balance' config
_CN.TRAINER.N_SAMPLES_PER_SUBSET = 200
_CN.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT = True  # whether sample each scene with replacement or not
_CN.TRAINER.SB_SUBSET_SHUFFLE = True  # after sampling from scenes, whether shuffle within the epoch or not
_CN.TRAINER.SB_REPEAT = 1  # repeat N times for training the sampled data
# 'random' config
_CN.TRAINER.RDM_REPLACEMENT = True
_CN.TRAINER.RDM_NUM_SAMPLES = None

# gradient clipping
_CN.TRAINER.GRADIENT_CLIPPING = 0.5

# reproducibility
# This seed affects the data sampling. With the same seed, the data sampling is promised
# to be the same. When resume training from a checkpoint, it's better to use a different
# seed, otherwise the sampled data will be exactly the same as before resuming, which will
# cause less unique data items sampled during the entire training.
# Use of different seed values might affect the final training result, since not all data items
# are used during training on ScanNet. (60M pairs of images sampled during traing from 230M pairs in total.)
_CN.TRAINER.SEED = 66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()


# ============================================
# ✅ 修改总结（Modification Summary）
# ============================================
"""
基于你的原始配置，只修改了以下参数以对齐CoMatch：

1. LOCAL_REGRESS_TEMPERATURE: 1.0 -> 10.0 (第46行)
2. LOCAL_REGRESS_SLICEDIM: 8 -> 64 (第47行，如果显存不足可改回8)
3. COARSE_OVERLAP_WEIGHT: False -> True (第59行)
4. FINE_OVERLAP_WEIGHT: False -> True (第60行)
5. FINE_TYPE: 'l2_with_std' -> 'l2' (第69行)
6. SIMILARITY_WEIGHT: 0.1 -> 0.0 (第82行，建议初期不使用)
7. CANONICAL_LR: 6e-3 -> 8e-3 (第175行)
8. WARMUP_RATIO: 0. -> 0.1 (第184行)
9. WARMUP_STEP: 4800 -> 1875 (第185行)
10. MSLR_MILESTONES: [3, 6, 9, 12] -> [8, 12, 16, 20, 24] (第190行)

保留了你的所有其他配置：
✓ BACKBONE_TYPE = 'RepVGG'
✓ 所有 COARSE 配置
✓ MAX_OFFSET_RATIO, ENABLE_BOUNDARY_CHECK, CONSISTENCY_THRESHOLD
✓ BILATERAL_WEIGHT = 1.0 (已有)
✓ 所有 DATASET 配置
✓ 所有其他 TRAINER 配置

如果显存不足，可以：
- 将 LOCAL_REGRESS_SLICEDIM 改回 8
- 减小 batch size
"""