#!/usr/bin/env python3
"""
测试HybridMamba的不同混合模式
"""

import torch
from HybridMamba import HybridMamba

def test_hybrid_modes():
    """测试不同的混合模式"""
    
    print("=" * 60)
    print("测试 HybridMamba 的不同混合模式")
    print("=" * 60)
    
    modes = ['alternate', 'mamba_first', 'attention_first', 'all_mamba', 'all_attention']
    
    for mode in modes:
        print(f"\n🔍 测试模式: {mode}")
        print("-" * 40)
        
        try:
            model = HybridMamba(
                in_chans=32,
                patch_size=200,
                out_dim=200,
                d_model=200,
                hybrid_mode=mode
            )
            
            # 测试前向传播
            x = torch.randn(2, 32, 4, 200)  # batch_size=2
            
            with torch.no_grad():
                output = model(x)
            
            print(f"✅ 模式 '{mode}' 测试成功")
            print(f"   输入形状: {x.shape}")
            print(f"   输出形状: {output.shape}")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   总参数量: {total_params:,}")
            
        except Exception as e:
            print(f"❌ 模式 '{mode}' 测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_hybrid_modes()