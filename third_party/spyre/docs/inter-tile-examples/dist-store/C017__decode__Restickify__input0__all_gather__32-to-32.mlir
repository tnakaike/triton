// C017 -- decode, Restickify input0, consumer bmm-wtAttnHeadBreak-VirtualReshape-Output-Restickify
// all_gather
// tensor bmm-wtAttnHeadBreak-VirtualReshape_out, rank 5 (j, mb, out, x, y), extents {j: 8, mb: 1, out: 128, x: 1, y: 768}, f16
//
// source      32 view(s) = 32 piece(s) x 1 owner, ct 0..31
// destination 32 view(s) = 32 piece(s) x 1 owner(s), ct 0..31
// composed    source 8x1x128x1x768, destination 8x1x128x1x768 = 8x1x128x1x768 x 1 slot(s) per piece
//
// Both sides composed, so the function states the whole movement and needs no
// launch table to be read. The destination view is the whole of what its views
// hold -- one slot per (piece, owner) -- so a core says which slot it writes:
//     j = ((tid // 1) % 4) * 2
//     mb = 0
//     out = ((tid // 4) % 2) * 64
//     x = 0
//     y = ((tid // 8) % 4) * 192
// Which cores take part is not stated: 32 of the 32 address a slot they
// own, and the slot's own ct_id is what says so. See README.md for the mapping
// and the provenance.

#src0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 23 >= 0)>
#src1 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 24 >= 0, -d4 + 47 >= 0)>
#src2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 240 >= 0, -d4 + 263 >= 0)>
#src3 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 264 >= 0, -d4 + 287 >= 0)>
#src4 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 288 >= 0, -d4 + 311 >= 0)>
#src5 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 312 >= 0, -d4 + 335 >= 0)>
#src6 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 336 >= 0, -d4 + 359 >= 0)>
#src7 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 360 >= 0, -d4 + 383 >= 0)>
#src8 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 407 >= 0)>
#src9 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 408 >= 0, -d4 + 431 >= 0)>
#src10 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 432 >= 0, -d4 + 455 >= 0)>
#src11 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 456 >= 0, -d4 + 479 >= 0)>
#src12 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 48 >= 0, -d4 + 71 >= 0)>
#src13 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 480 >= 0, -d4 + 503 >= 0)>
#src14 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 504 >= 0, -d4 + 527 >= 0)>
#src15 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 528 >= 0, -d4 + 551 >= 0)>
#src16 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 552 >= 0, -d4 + 575 >= 0)>
#src17 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 599 >= 0)>
#src18 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 600 >= 0, -d4 + 623 >= 0)>
#src19 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 624 >= 0, -d4 + 647 >= 0)>
#src20 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 648 >= 0, -d4 + 671 >= 0)>
#src21 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 672 >= 0, -d4 + 695 >= 0)>
#src22 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 696 >= 0, -d4 + 719 >= 0)>
#src23 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 72 >= 0, -d4 + 95 >= 0)>
#src24 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 720 >= 0, -d4 + 743 >= 0)>
#src25 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 744 >= 0, -d4 + 767 >= 0)>
#src26 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 96 >= 0, -d4 + 119 >= 0)>
#src27 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 120 >= 0, -d4 + 143 >= 0)>
#src28 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 144 >= 0, -d4 + 167 >= 0)>
#src29 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 168 >= 0, -d4 + 191 >= 0)>
#src30 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 215 >= 0)>
#src31 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 216 >= 0, -d4 + 239 >= 0)>
#dst0 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst1 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst2 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst3 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst4 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst5 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst6 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst7 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst8 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst9 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst10 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst11 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst12 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst13 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst14 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst15 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst16 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 384 >= 0, -d4 + 575 >= 0)>
#dst17 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst18 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst19 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst20 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst21 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst22 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst23 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst24 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst25 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 - 576 >= 0, -d4 + 767 >= 0)>
#dst26 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst27 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst28 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 4 >= 0, -d0 + 5 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst29 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 6 >= 0, -d0 + 7 >= 0, d1 >= 0, -d1 >= 0, d2 - 64 >= 0, -d2 + 127 >= 0, d3 >= 0, -d3 >= 0, d4 >= 0, -d4 + 191 >= 0)>
#dst30 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#dst31 = affine_set<(d0, d1, d2, d3, d4) : (d0 - 2 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 >= 0, d4 - 192 >= 0, -d4 + 383 >= 0)>
#ord0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @c017_decode_restickify_input0(%src: index, %dst: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  %tid = ktdp.get_compute_tile_id : index

  // The source distribution: one view per partition, differing ONLY in
  // coordinate_set and ct_id.
  %s0 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %s1 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %s2 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %s3 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %s4 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %s5 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %s6 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %s7 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %s8 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %s9 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %s10 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %s11 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %s12 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %s13 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %s14 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %s15 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %s16 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %s17 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %s18 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %s19 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %s20 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %s21 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %s22 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %s23 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %s24 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %s25 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %s26 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %s27 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %s28 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %s29 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %s30 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %s31 = ktdp.construct_memory_view %src, sizes: [8, 1, 128, 1, 24], strides: [3072, 3072, 24, 24, 1]
      {coordinate_set = #src31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %from = ktdp.construct_distributed_memory_view (%s0, %s1, %s2, %s3, %s4, %s5, %s6, %s7, %s8, %s9, %s10, %s11, %s12, %s13, %s14, %s15, %s16, %s17, %s18, %s19, %s20, %s21, %s22, %s23, %s24, %s25, %s26, %s27, %s28, %s29, %s30, %s31
      : memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<8x1x128x1x24xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<8x1x128x1x768xf16>

  // The destination distribution, the same way. A piece with several owners
  // is that many views with the SAME coordinate_set and different ct_id --
  // which is what replication is, stated rather than left to the launch. The
  // composed type counts those views, so the slots are distinct coordinates
  // and no coordinate has two writers.
  %d0 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst0, memory_space = #ktdp.memory_space<ct_local, ct_id = 0>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 0>>
  %d1 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst1, memory_space = #ktdp.memory_space<ct_local, ct_id = 1>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 1>>
  %d2 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst2, memory_space = #ktdp.memory_space<ct_local, ct_id = 10>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 10>>
  %d3 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst3, memory_space = #ktdp.memory_space<ct_local, ct_id = 11>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 11>>
  %d4 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst4, memory_space = #ktdp.memory_space<ct_local, ct_id = 12>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 12>>
  %d5 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst5, memory_space = #ktdp.memory_space<ct_local, ct_id = 13>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 13>>
  %d6 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst6, memory_space = #ktdp.memory_space<ct_local, ct_id = 14>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 14>>
  %d7 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst7, memory_space = #ktdp.memory_space<ct_local, ct_id = 15>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 15>>
  %d8 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst8, memory_space = #ktdp.memory_space<ct_local, ct_id = 16>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 16>>
  %d9 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst9, memory_space = #ktdp.memory_space<ct_local, ct_id = 17>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 17>>
  %d10 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst10, memory_space = #ktdp.memory_space<ct_local, ct_id = 18>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 18>>
  %d11 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst11, memory_space = #ktdp.memory_space<ct_local, ct_id = 19>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 19>>
  %d12 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst12, memory_space = #ktdp.memory_space<ct_local, ct_id = 2>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 2>>
  %d13 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst13, memory_space = #ktdp.memory_space<ct_local, ct_id = 20>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 20>>
  %d14 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst14, memory_space = #ktdp.memory_space<ct_local, ct_id = 21>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 21>>
  %d15 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst15, memory_space = #ktdp.memory_space<ct_local, ct_id = 22>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 22>>
  %d16 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst16, memory_space = #ktdp.memory_space<ct_local, ct_id = 23>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 23>>
  %d17 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst17, memory_space = #ktdp.memory_space<ct_local, ct_id = 24>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 24>>
  %d18 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst18, memory_space = #ktdp.memory_space<ct_local, ct_id = 25>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 25>>
  %d19 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst19, memory_space = #ktdp.memory_space<ct_local, ct_id = 26>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 26>>
  %d20 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst20, memory_space = #ktdp.memory_space<ct_local, ct_id = 27>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 27>>
  %d21 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst21, memory_space = #ktdp.memory_space<ct_local, ct_id = 28>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 28>>
  %d22 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst22, memory_space = #ktdp.memory_space<ct_local, ct_id = 29>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 29>>
  %d23 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst23, memory_space = #ktdp.memory_space<ct_local, ct_id = 3>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 3>>
  %d24 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst24, memory_space = #ktdp.memory_space<ct_local, ct_id = 30>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 30>>
  %d25 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst25, memory_space = #ktdp.memory_space<ct_local, ct_id = 31>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 31>>
  %d26 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst26, memory_space = #ktdp.memory_space<ct_local, ct_id = 4>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 4>>
  %d27 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst27, memory_space = #ktdp.memory_space<ct_local, ct_id = 5>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 5>>
  %d28 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst28, memory_space = #ktdp.memory_space<ct_local, ct_id = 6>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 6>>
  %d29 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst29, memory_space = #ktdp.memory_space<ct_local, ct_id = 7>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 7>>
  %d30 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst30, memory_space = #ktdp.memory_space<ct_local, ct_id = 8>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 8>>
  %d31 = ktdp.construct_memory_view %dst, sizes: [2, 1, 64, 1, 192], strides: [12288, 12288, 192, 192, 1]
      {coordinate_set = #dst31, memory_space = #ktdp.memory_space<ct_local, ct_id = 9>}
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 9>>

  %to = ktdp.construct_distributed_memory_view (%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23, %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31
      : memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 0>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 1>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 10>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 11>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 12>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 13>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 14>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 15>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 16>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 17>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 18>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 19>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 2>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 20>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 21>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 22>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 23>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 24>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 25>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 26>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 27>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 28>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 29>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 3>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 30>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 31>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 4>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 5>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 6>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 7>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 8>>, memref<2x1x64x1x192xf16, #ktdp.memory_space<ct_local, ct_id = 9>>) : memref<8x1x128x1x768xf16>

  // Where this core's box sits: one independent field of the tile id per
  // divided dimension, exactly as in ../if-store/. No branch selects it --
  // the index is computed.
  %i_j = arith.remsi %tid, %c4 : index
  %o_j = arith.muli %i_j, %c2 : index
  %q_out = arith.divsi %tid, %c4 : index
  %i_out = arith.remsi %q_out, %c2 : index
  %o_out = arith.muli %i_out, %c64 : index
  %q_y = arith.divsi %tid, %c8 : index
  %i_y = arith.remsi %q_y, %c4 : index
  %o_y = arith.muli %i_y, %c192 : index

  // The movement, and no control flow: one load of a coordinate REGION --
  // which source partitions that touches is the composed view's to resolve,
  // and a region may span several -- and one store into a named slot of the
  // destination, whose ct_id is what says whether this core owns it.
  %rt = ktdp.construct_access_tile %from[%o_j, %c0, %o_out, %c0, %o_y]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<8x1x128x1x768xf16> -> !ktdp.access_tile<2x1x64x1x192xindex>
  %val = ktdp.load %rt : <2x1x64x1x192xindex> -> tensor<2x1x64x1x192xf16>
  %wt = ktdp.construct_access_tile %to[%o_j, %c0, %o_out, %c0, %o_y]
      {access_tile_order = #ord0, access_tile_set = #dst0}
      : memref<8x1x128x1x768xf16> -> !ktdp.access_tile<2x1x64x1x192xindex>
  ktdp.store %val, %wt : tensor<2x1x64x1x192xf16>, <2x1x64x1x192xindex>

  return
}
