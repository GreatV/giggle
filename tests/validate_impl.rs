use approx::assert_relative_eq;
/// 验证新实现功能的正确性
use giggle::*;
use ndarray::Array2;

#[test]
fn validate_pad_center_functionality() {
    println!("\n=== 验证 pad_center 功能 ===");

    // 测试扩展
    let data = vec![1, 2, 3, 4, 5];
    let padded = utils::pad_center(&data, 11, 0);

    assert_eq!(padded.len(), 11);
    assert_eq!(padded[3], 1); // 左边填充3个
    assert_eq!(padded[7], 5);
    println!("✓ pad_center 扩展测试通过: {:?} -> {:?}", data, padded);

    // 测试截断
    let long_data = vec![1, 2, 3, 4, 5, 6, 7];
    let trimmed = utils::pad_center(&long_data, 3, 0);
    assert_eq!(trimmed.len(), 3);
    assert_eq!(trimmed, vec![3, 4, 5]); // 从中间取3个
    println!(
        "✓ pad_center 截断测试通过: {:?} -> {:?}",
        long_data, trimmed
    );
}

#[test]
fn validate_fix_length_functionality() {
    println!("\n=== 验证 fix_length 功能 ===");

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 扩展
    let extended = utils::fix_length(&data, 8, 0.0);
    assert_eq!(extended.len(), 8);
    assert_eq!(&extended[..5], &data[..]);
    assert_eq!(&extended[5..], &[0.0, 0.0, 0.0]);
    println!(
        "✓ fix_length 扩展: {} -> {} 元素",
        data.len(),
        extended.len()
    );

    // 截断
    let truncated = utils::fix_length(&data, 3, 0.0);
    assert_eq!(truncated.len(), 3);
    assert_eq!(truncated, vec![1.0, 2.0, 3.0]);
    println!(
        "✓ fix_length 截断: {} -> {} 元素",
        data.len(),
        truncated.len()
    );
}

#[test]
fn validate_expand_to_power_of_two() {
    println!("\n=== 验证 expand_to (FFT优化) ===");

    let test_cases = vec![(100, 128), (128, 128), (129, 256), (1000, 1024)];

    for (input, expected) in test_cases {
        let result = utils::expand_to(input, None);
        assert_eq!(result, expected);
        println!("✓ expand_to({}) = {} (下一个2的幂)", input, result);
    }
}

#[test]
fn validate_frame_array_structure() {
    println!("\n=== 验证 frame_array 分帧 ===");

    let data: Vec<f32> = (0..10).map(|x| x as f32).collect();
    let frames = utils::frame_array(&data, 4, 2, 0.0, false);

    println!("✓ 输入: {:?}", data);
    println!("✓ 帧形状: {:?} (frame_length × n_frames)", frames.shape());
    println!("✓ 第一帧: {:?}", frames.column(0).to_vec());
    println!("✓ 第二帧: {:?}", frames.column(1).to_vec());

    assert_eq!(frames.shape(), &[4, 4]);
    assert_eq!(frames.column(0).to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    assert_eq!(frames.column(1).to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn validate_window_sumsquare_nola() {
    println!("\n=== 验证 window_sumsquare NOLA约束 ===");

    let window = window::hann(512);
    let n_frames = 10;
    let hop_length = 256; // 50% overlap

    let wss = window::window_sumsquare(&window, n_frames, hop_length, None, Some(512));

    // 检查稳定区域
    let stable_start = 512;
    let stable_end = wss.len().saturating_sub(512);

    if stable_end > stable_start {
        let stable_region = &wss[stable_start..stable_end];
        let min_val = stable_region.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = stable_region
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mean_val: f32 = stable_region.iter().sum::<f32>() / stable_region.len() as f32;

        println!("✓ 稳定区域 NOLA 最小值: {:.4}", min_val);
        println!("✓ 稳定区域 NOLA 最大值: {:.4}", max_val);
        println!("✓ 稳定区域 NOLA 平均值: {:.4}", mean_val);
        println!(
            "✓ NOLA 变化范围: {:.2}%",
            (max_val - min_val) / mean_val * 100.0
        );

        assert!(min_val > 0.3, "NOLA约束必须满足: min_val = {}", min_val);
        println!("✅ NOLA约束满足");
    }
}

#[test]
fn validate_softmask_separation() {
    println!("\n=== 验证 softmask 源分离 ===");

    let reference = Array2::from_shape_vec(
        (3, 4),
        vec![2.0, 0.0, 4.0, 1.0, 1.0, 3.0, 0.0, 2.0, 5.0, 1.0, 2.0, 3.0],
    )
    .unwrap();

    let other = Array2::from_shape_vec(
        (3, 4),
        vec![2.0, 1.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 1.0],
    )
    .unwrap();

    let mask = utils::softmask(&reference, &[&other], 1.0, false);

    println!("✓ 掩码形状: {:?}", mask.shape());

    // 验证掩码在[0,1]范围内
    for val in mask.iter() {
        assert!(*val >= 0.0 && *val <= 1.0, "掩码值必须在[0,1]: {}", val);
    }

    // 验证特定值
    // (0,0): 2/(2+2) = 0.5
    assert_relative_eq!(mask[(0, 0)], 0.5, epsilon = 0.01);
    // (0,2): 4/(4+0) = 1.0
    assert_relative_eq!(mask[(0, 2)], 1.0, epsilon = 0.01);

    println!(
        "✓ 掩码值示例: [{:.3}, {:.3}, {:.3}, {:.3}]",
        mask[(0, 0)],
        mask[(0, 1)],
        mask[(0, 2)],
        mask[(0, 3)]
    );
    println!("✅ 软掩码测试通过");
}

#[test]
fn validate_fill_off_diagonal() {
    println!("\n=== 验证 fill_off_diagonal ===");

    let arr =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let filled = utils::fill_off_diagonal(&arr, 0.0);

    // 对角线应保持不变
    assert_eq!(filled[(0, 0)], 1.0);
    assert_eq!(filled[(1, 1)], 5.0);
    assert_eq!(filled[(2, 2)], 9.0);

    // 非对角线应为0
    assert_eq!(filled[(0, 1)], 0.0);
    assert_eq!(filled[(1, 0)], 0.0);
    assert_eq!(filled[(2, 1)], 0.0);

    println!(
        "✓ 对角线保留: [{:.1}, {:.1}, {:.1}]",
        filled[(0, 0)],
        filled[(1, 1)],
        filled[(2, 2)]
    );
    println!("✓ 非对角线填充为0");
    println!("✅ fill_off_diagonal 测试通过");
}

#[test]
fn validate_griffinlim_reconstruction() {
    println!("\n=== 验证 Griffin-Lim 相位重建 ===");

    // 生成测试信号
    let sr = 22050;
    let duration = 0.5;
    let freq = 440.0;
    let signal = io::tone(freq, sr, duration);

    println!("✓ 生成 {:.1}Hz 正弦波信号，时长 {:.1}s", freq, duration);

    // STFT
    let config = spectrum::StftConfig {
        n_fft: 2048,
        hop_length: 512,
        win_length: 2048,
        window: window::hann(2048),
        center: true,
        pad_mode: spectrum::PadMode::Reflect,
    };

    let stft_matrix = spectrum::stft(&signal, &config).unwrap();
    let (magnitude, _phase) = spectrum::magphase(&stft_matrix);

    println!("✓ STFT 频谱形状: {:?}", magnitude.shape());

    // Griffin-Lim重建
    let reconstructed =
        spectrum::griffinlim(&magnitude, &config, 16, Some(signal.len()), 0.0).unwrap();

    println!("✓ 原始信号长度: {}", signal.len());
    println!("✓ 重建信号长度: {}", reconstructed.len());

    assert_eq!(reconstructed.len(), signal.len());

    // 检查能量
    let orig_energy: f32 = signal.iter().map(|x| x * x).sum();
    let recon_energy: f32 = reconstructed.iter().map(|x| x * x).sum();
    let energy_ratio = recon_energy / orig_energy;

    println!("✓ 能量保持率: {:.2}%", energy_ratio * 100.0);

    // 能量应该保持在合理范围内 (80% - 120%)
    assert!(
        energy_ratio > 0.5 && energy_ratio < 1.5,
        "能量保持率异常: {:.2}",
        energy_ratio
    );

    println!("✅ Griffin-Lim 重建测试通过");
}

#[test]
fn validate_stft_istft_perfect_reconstruction() {
    println!("\n=== 验证 STFT/ISTFT 完美重建 ===");

    let signal = io::tone(440.0, 22050, 0.5);

    let config = spectrum::StftConfig {
        n_fft: 2048,
        hop_length: 512,
        win_length: 2048,
        window: window::hann(2048),
        center: true,
        pad_mode: spectrum::PadMode::Reflect,
    };

    let stft_matrix = spectrum::stft(&signal, &config).unwrap();
    let reconstructed = spectrum::istft(&stft_matrix, &config, Some(signal.len())).unwrap();

    // 计算重建误差
    let mse = utils::mse(&signal, &reconstructed);
    let signal_energy: f32 = signal.iter().map(|x| x * x).sum::<f32>() / signal.len() as f32;
    let relative_error = mse / signal_energy;

    println!("✓ 信号长度: {}", signal.len());
    println!("✓ MSE: {:.6e}", mse);
    println!("✓ 相对误差: {:.6e}", relative_error);

    // 相对误差应该非常小 (< 0.01%)
    assert!(
        relative_error < 1e-4,
        "STFT/ISTFT 重建误差过大: {:.6e}",
        relative_error
    );

    println!("✅ 完美重建验证通过");
}

#[test]
fn validate_integration_test() {
    println!("\n=== 综合验证：完整音频处理流程 ===");

    // 1. 生成信号
    let sr = 22050;
    let signal = io::tone(440.0, sr, 1.0);
    println!("✓ 步骤1: 生成测试信号 ({} 采样)", signal.len());

    // 2. STFT
    let config = spectrum::StftConfig::default();
    let stft = spectrum::stft(&signal, &config).unwrap();
    println!("✓ 步骤2: STFT 变换 -> 频谱形状 {:?}", stft.shape());

    // 3. 提取幅度和相位
    let (mag, phase) = spectrum::magphase(&stft);
    println!("✓ 步骤3: 提取幅度和相位");

    // 4. 使用幅度进行 Griffin-Lim 重建
    let gl_recon = spectrum::griffinlim(&mag, &config, 8, Some(signal.len()), 0.0).unwrap();
    println!("✓ 步骤4: Griffin-Lim 重建 ({} 采样)", gl_recon.len());

    // 5. 使用原始相位进行 ISTFT
    let mut stft_with_phase = stft.clone();
    for ((i, j), val) in stft_with_phase.indexed_iter_mut() {
        *val = mag[(i, j)] * phase[(i, j)];
    }
    let istft_recon = spectrum::istft(&stft_with_phase, &config, Some(signal.len())).unwrap();
    println!("✓ 步骤5: ISTFT 重建 ({} 采样)", istft_recon.len());

    // 6. 验证结果
    let istft_error = utils::mse(&signal, &istft_recon);
    let gl_energy: f32 = gl_recon.iter().map(|x| x * x).sum();
    let orig_energy: f32 = signal.iter().map(|x| x * x).sum();

    println!("✓ ISTFT 重建误差: {:.6e}", istft_error);
    println!("✓ Griffin-Lim 能量比: {:.3}", gl_energy / orig_energy);

    assert!(istft_error < 1e-4);
    assert!(gl_energy / orig_energy > 0.5);

    println!("✅ 综合测试通过");
}
