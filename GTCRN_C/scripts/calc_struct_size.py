#!/usr/bin/env python3
"""
Calculate expected C struct size for GTCRN weights.
"""

def calc_struct_size():
    """Calculate expected size of gtcrn_weights_s struct."""
    sizes = {}
    total = 0

    # ERB filterbank
    sizes['erb_fc_weight'] = 64 * 192  # 12288
    sizes['ierb_fc_weight'] = 192 * 64  # 12288

    # Encoder ConvBlock 0
    sizes['en_conv0_weight'] = 16 * 9 * 1 * 5  # 720
    sizes['en_conv0_bias'] = 16
    sizes['en_bn0_gamma'] = 16
    sizes['en_bn0_beta'] = 16
    sizes['en_bn0_mean'] = 16
    sizes['en_bn0_var'] = 16
    sizes['en_prelu0'] = 1

    # Encoder ConvBlock 1
    sizes['en_conv1_weight'] = 16 * 8 * 1 * 5  # 640
    sizes['en_conv1_bias'] = 16
    sizes['en_bn1_gamma'] = 16
    sizes['en_bn1_beta'] = 16
    sizes['en_bn1_mean'] = 16
    sizes['en_bn1_var'] = 16
    sizes['en_prelu1'] = 1

    # GTConvBlock helper function
    def gtconvblock_size(prefix, is_decoder=False):
        block = {}
        block[f'{prefix}_pc1_weight'] = 16 * 24 * 1 * 1 if not is_decoder else 24 * 16 * 1 * 1  # 384
        block[f'{prefix}_pc1_bias'] = 16
        block[f'{prefix}_bn1_gamma'] = 16
        block[f'{prefix}_bn1_beta'] = 16
        block[f'{prefix}_bn1_mean'] = 16
        block[f'{prefix}_bn1_var'] = 16
        block[f'{prefix}_prelu1'] = 1

        block[f'{prefix}_dc_weight'] = 16 * 1 * 3 * 3  # 144
        block[f'{prefix}_dc_bias'] = 16
        block[f'{prefix}_bn2_gamma'] = 16
        block[f'{prefix}_bn2_beta'] = 16
        block[f'{prefix}_bn2_mean'] = 16
        block[f'{prefix}_bn2_var'] = 16
        block[f'{prefix}_prelu2'] = 1

        block[f'{prefix}_pc2_weight'] = 8 * 16 * 1 * 1 if not is_decoder else 16 * 8 * 1 * 1  # 128
        block[f'{prefix}_pc2_bias'] = 8
        block[f'{prefix}_bn3_gamma'] = 8
        block[f'{prefix}_bn3_beta'] = 8
        block[f'{prefix}_bn3_mean'] = 8
        block[f'{prefix}_bn3_var'] = 8

        # TRA
        block[f'{prefix}_tra_gru_ih'] = 48 * 8  # 384
        block[f'{prefix}_tra_gru_hh'] = 48 * 16  # 768
        block[f'{prefix}_tra_gru_bih'] = 48
        block[f'{prefix}_tra_gru_bhh'] = 48
        block[f'{prefix}_tra_fc_weight'] = 8 * 16  # 128
        block[f'{prefix}_tra_fc_bias'] = 8

        return block

    # Encoder GTConvBlocks
    for i in [2, 3, 4]:
        block = gtconvblock_size(f'en_gt{i}')
        sizes.update(block)

    # DPGRNN helper
    def dpgrnn_size(prefix):
        dp = {}
        # Intra RNN (bidirectional)
        for rnn in ['rnn1', 'rnn2']:
            dp[f'{prefix}_intra_{rnn}_ih'] = 12 * 8  # 96
            dp[f'{prefix}_intra_{rnn}_hh'] = 12 * 4  # 48
            dp[f'{prefix}_intra_{rnn}_bih'] = 12
            dp[f'{prefix}_intra_{rnn}_bhh'] = 12
            dp[f'{prefix}_intra_{rnn}_ih_rev'] = 12 * 8
            dp[f'{prefix}_intra_{rnn}_hh_rev'] = 12 * 4
            dp[f'{prefix}_intra_{rnn}_bih_rev'] = 12
            dp[f'{prefix}_intra_{rnn}_bhh_rev'] = 12
        dp[f'{prefix}_intra_fc_weight'] = 16 * 16  # 256
        dp[f'{prefix}_intra_fc_bias'] = 16
        dp[f'{prefix}_intra_ln_gamma'] = 33 * 16  # 528
        dp[f'{prefix}_intra_ln_beta'] = 33 * 16  # 528

        # Inter RNN (unidirectional)
        for rnn in ['rnn1', 'rnn2']:
            dp[f'{prefix}_inter_{rnn}_ih'] = 24 * 8  # 192
            dp[f'{prefix}_inter_{rnn}_hh'] = 24 * 8  # 192
            dp[f'{prefix}_inter_{rnn}_bih'] = 24
            dp[f'{prefix}_inter_{rnn}_bhh'] = 24
        dp[f'{prefix}_inter_fc_weight'] = 16 * 16
        dp[f'{prefix}_inter_fc_bias'] = 16
        dp[f'{prefix}_inter_ln_gamma'] = 33 * 16
        dp[f'{prefix}_inter_ln_beta'] = 33 * 16

        return dp

    sizes.update(dpgrnn_size('dp1'))
    sizes.update(dpgrnn_size('dp2'))

    # Decoder GTConvBlocks
    for i in [0, 1, 2]:
        block = gtconvblock_size(f'de_gt{i}', is_decoder=True)
        sizes.update(block)

    # Decoder ConvBlock 3
    sizes['de_conv3_weight'] = 16 * 8 * 1 * 5  # 640
    sizes['de_conv3_bias'] = 16
    sizes['de_bn3_gamma'] = 16
    sizes['de_bn3_beta'] = 16
    sizes['de_bn3_mean'] = 16
    sizes['de_bn3_var'] = 16
    sizes['de_prelu3'] = 1

    # Decoder ConvBlock 4
    sizes['de_conv4_weight'] = 16 * 2 * 1 * 5  # 160
    sizes['de_conv4_bias'] = 2
    sizes['de_bn4_gamma'] = 2
    sizes['de_bn4_beta'] = 2
    sizes['de_bn4_mean'] = 2
    sizes['de_bn4_var'] = 2

    total = sum(sizes.values())

    print("=" * 60)
    print("C Struct Size Calculation")
    print("=" * 60)

    # Print by section
    print("\nERB filterbank:")
    erb = sum(v for k, v in sizes.items() if k.startswith('erb'))
    print(f"  Total: {erb}")

    print("\nEncoder ConvBlocks 0-1:")
    en_conv = sum(v for k, v in sizes.items() if k.startswith('en_conv') or k.startswith('en_bn') or k.startswith('en_prelu'))
    print(f"  Total: {en_conv}")

    print("\nEncoder GTConvBlocks 2-4:")
    en_gt = sum(v for k, v in sizes.items() if k.startswith('en_gt'))
    print(f"  Total: {en_gt}")

    print("\nDPGRNN 1 + 2:")
    dp = sum(v for k, v in sizes.items() if k.startswith('dp'))
    print(f"  Total: {dp}")

    print("\nDecoder GTConvBlocks 0-2:")
    de_gt = sum(v for k, v in sizes.items() if k.startswith('de_gt'))
    print(f"  Total: {de_gt}")

    print("\nDecoder ConvBlocks 3-4:")
    de_conv = sum(v for k, v in sizes.items() if k.startswith('de_conv') or k.startswith('de_bn') or k.startswith('de_prelu'))
    print(f"  Total: {de_conv}")

    print("\n" + "=" * 60)
    print(f"Total floats in C struct: {total}")
    print(f"Total bytes (floats * 4): {total * 4}")
    print(f"With 8-byte header: {total * 4 + 8}")
    print("=" * 60)

    return total


if __name__ == "__main__":
    calc_struct_size()
