#!/bin/bash
#SBATCH --job-name=ms_rs_exp4
#SBATCH --partition=gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/data/home/scxj642/.conda/envs/ms-rs_env/lib:${LD_LIBRARY_PATH:-}

EXP_NAME="7"
LOG_DIR="/data/home/scxj642/run/yujy/MS-RS/results_experiment/exp_5x10x20x_CN_config"
N_FOLDS=0
SCALE_CONFIGS='{
        "5x": {"gcn_hidden": 512, "gcn_out": 128, "k_neighbors": 8, "gcn_layers": 2, "dropout": 0.3, "temperature": 1.2, "lr": 0.00012},
        "10x": {"gcn_hidden": 512, "gcn_out": 256, "k_neighbors": 8, "gcn_layers": 2, "dropout": 0.4, "temperature": 0.8, "lr": 0.00025},
        "20x": {"gcn_hidden": 512, "gcn_out": 512, "k_neighbors": 12, "gcn_layers": 2, "dropout": 0.5, "temperature": 0.7, "lr": 0.00045},
        "clinical": {"lr": 0.0002}
}'

# ============================================================
# ✅ 清理临时文件函数
# ============================================================
cleanup_temp() {
    echo "🧹 清理临时文件..."
    find /tmp -maxdepth 1 -name "*.tmp" -user "$(whoami)" -delete 2>/dev/null
    find /tmp -maxdepth 1 -name "torch_*" -user "$(whoami)" -exec rm -rf {} + 2>/dev/null
    # 清理可能的缓存目录
    rm -rf /tmp/scxj642* 2>/dev/null
    echo "🧹 临时文件清理完成。"
}

# ============================================================
# ✅ 错误检测和重试函数
# ============================================================
run_with_retry() {
    local cmd="$1"
    local max_retries=3
    local retry_count=0
    local exit_code=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo ""
        if [ $retry_count -gt 0 ]; then
            echo "🔄 第 ${retry_count} 次重试..."
        fi
        
        # 运行命令
        eval "$cmd"
        exit_code=$?
        
        # 检查是否是I/O错误
        if [ $exit_code -ne 0 ]; then
            # 检查错误信息中是否包含I/O相关错误
            if [ -f "${LOG_DIR}/${EXP_NAME}/fold_${FOLD}/train.log" ]; then
                if grep -q "Input/output error" "${LOG_DIR}/${EXP_NAME}/fold_${FOLD}/train.log" 2>/dev/null; then
                    echo "⚠️ 检测到 I/O 错误！"
                    echo "🧹 清理缓存并准备重试..."
                    cleanup_temp
                    retry_count=$((retry_count + 1))
                    if [ $retry_count -lt $max_retries ]; then
                        echo "⏳ 等待 2 分钟后重试... (${retry_count}/${max_retries})"
                        sleep 120
                        continue
                    fi
                fi
            fi
            # 非I/O错误或其他情况
            if [ $retry_count -ge $max_retries ]; then
                echo "❌ 达到最大重试次数 (${max_retries})，放弃重试"
            fi
            break
        else
            # 成功执行
            break
        fi
    done
    
    return $exit_code
}

# ============================================================
# ✅ 单折训练函数（带重试）
# ============================================================
run_fold_with_retry() {
    local fold=$1
    local max_attempts=3
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if [ $attempt -gt 0 ]; then
            echo ""
            echo "🔄 第 ${fold} 折 - 第 ${attempt} 次重试..."
            cleanup_temp
            echo "⏳ 等待 2 分钟后重试..."
            sleep 120
        fi
        
        /data/home/scxj642/.conda/envs/ms-rs_env/bin/python \
            "/data/home/scxj642/run/yujy/MS-RS_github/train/MAIN.py" \
            --name                        "${EXP_NAME}" \
            --device                      cuda \
            --EPOCH                       200 \
            --magnifications              5x 10x 20x \
            --batch_size                  4 \
            --patience                    15 \
            --min_epochs                  80 \
            --lr                          0.00065 \
            --warmup_epochs               20 \
            --warmup_lr                   0.000325 \
            --min_lr                      0.0001 \
            --dropout                     0.5 \
            --weight_decay                0.00035 \
            --use_contrastive             True \
            --use_instance_branch         True \
            --contrastive_weight          0.15 \
            --instance_weight             0.1 \
            --contrast_temperature        0.7 \
            --proj_dim                    256 \
            --memory_bank_size            128 \
            --contrastive_ramp_epochs     10 \
            --in_dim                      768 \
            --gcn_hidden                  512 \
            --focal_gamma                 1.5 \
            --use_focal_loss              True \
            --label_smoothing             0.12 \
            --use_pseudo_bag_aug          False \
            --pseudo_bag_ratio            3.0 \
            --pseudo_bag_sample_ratio     0.75 \
            --pseudo_bag_mix_prob         0.3 \
            --pseudo_bag_noise_std        0.02 \
            --use_undersampling           False \
            --use_weighted_sampling       False \
            --weight_strategy             temperature \
            --sampling_temperature        1.2 \
            --sampling_smooth_factor      0.5 \
            --dynamic_sampling_warmup     20 \
            --gradient_accumulation_steps 12 \
            --weight_regularization       0.02 \
            --weight_lr_multiplier        3.0 \
            --ensemble_lr_multiplier      0.95 \
            --use_dynamic_threshold       True \
            --threshold_search_metric     youden \
            --run_fold                    "$fold" \
            --n_folds                     "${N_FOLDS}" \
            --n_workers                   1 \
            --ensemble_method             attention \
            --features_dir                /data/home/scxj642/run/yujy/All_data/features \
            --csv_dir                     /data/home/scxj642/run/yujy/MS-RS/csv_data/5fold \
            --log_dir                     "${LOG_DIR}" \
            --use_clinical                True \
            --fusion_method               attention \
            --clinical_csv                /data/home/scxj642/run/yujy/MS-RS/csv_data/all_pca.csv \
            --clinical_hidden             256 \
            --subtype_embed_dim           128 \
            --clinical_dropout            0.3 \
            --scale_configs               "${SCALE_CONFIGS}" \
            2>&1 | tee "${LOG_DIR}/${EXP_NAME}/fold_${fold}/train.log"
        
        local exit_code=$?
        
        # 检查是否是I/O错误
        if [ $exit_code -ne 0 ]; then
            if grep -q "Input/output error" "${LOG_DIR}/${EXP_NAME}/fold_${fold}/train.log" 2>/dev/null; then
                echo "⚠️ 第 ${fold} 折遇到 I/O 错误！"
                attempt=$((attempt + 1))
                if [ $attempt -lt $max_attempts ]; then
                    echo "🔄 准备重试第 ${fold} 折... (${attempt}/${max_attempts})"
                    continue
                else
                    echo "❌ 第 ${fold} 折失败，已达到最大重试次数"
                    return $exit_code
                fi
            else
                # 非I/O错误，直接返回
                return $exit_code
            fi
        else
            # 成功
            return 0
        fi
    done
}

mkdir -p "${LOG_DIR}/${EXP_NAME}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${EXP_NAME}  GPU=${CUDA_VISIBLE_DEVICES}"

# 实验开始前清理一次
cleanup_temp

for (( FOLD=0; FOLD<=$N_FOLDS; FOLD++ )); do
    echo ""
    echo "========================================"
    echo "📊 开始第 ${FOLD} 折训练"
    echo "========================================"
    mkdir -p "${LOG_DIR}/${EXP_NAME}/fold_${FOLD}"

    # 每折开始前清理一次
    cleanup_temp

    # 运行当前折（带重试机制）
    run_fold_with_retry "${FOLD}"
    fold_exit_code=$?
    
    if [ $fold_exit_code -eq 0 ]; then
        echo "✅ 第 ${FOLD} 折训练完成"
    else
        echo "❌ 第 ${FOLD} 折训练失败（已重试）"
        # 即使失败也继续下一折
    fi

    # 每折结束后清理
    cleanup_temp

    # 最后一折结束后不需要等待，否则等待 2 分钟
    if (( FOLD < N_FOLDS - 1 )); then
        echo "⏳ 等待 2 分钟后开始下一折..."
        sleep 120
    fi
done

# 所有折完成后最后清理一次
echo ""
echo "========================================"
echo "🎉 实验 ${EXP_NAME} 全部完成！"
echo "========================================"
cleanup_temp

echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  ${EXP_NAME}"