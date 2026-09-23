use icy_sixel::{EncodeOptions, QuantizeMethod, SixelEncoder, SixelError, SixelImage};

#[test]
fn long_runs_roundtrip_across_repeat_limit() {
    for width in [65_535, 65_536, 131_073] {
        let pixels = [255, 0, 0, 255].repeat(width);
        let image = SixelImage::try_from_rgba(pixels.clone(), width, 1).unwrap();
        let encoded = image.encode().unwrap();
        let decoded = SixelImage::decode(encoded.as_bytes()).unwrap();

        assert_eq!(decoded.width, width);
        assert_eq!(&decoded.pixels[..pixels.len()], pixels.as_slice());
        for run in encoded.split('!').skip(1) {
            let count: usize = run.chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap();
            assert!(count <= 65_535);
        }
    }
}

#[test]
fn overflowing_dimensions_return_errors() {
    // Exercise both multiplication steps, even through the unchecked constructor.
    for (width, height) in [(usize::MAX, 2), (usize::MAX / 4 + 1, 1)] {
        let image = SixelImage::from_rgba(Vec::new(), width, height);
        assert!(matches!(image.encode(), Err(SixelError::IntegerOverflow)));
        assert!(matches!(image.encode_with(&EncodeOptions::default()), Err(SixelError::IntegerOverflow)));
        assert!(matches!(
            icy_sixel::sixel_encode(&[], width, height, &EncodeOptions::default()),
            Err(SixelError::IntegerOverflow)
        ));
    }
}

#[test]
fn transparent_pixels_do_not_consume_palette_colors() {
    for quantize_method in [QuantizeMethod::Wu, QuantizeMethod::kmeans()] {
        for diffusion in [0.0, 0.875, 1.0] {
            let mut pixels = [0, 255, 0, 127].repeat(1000);
            pixels.extend_from_slice(&[255, 0, 0, 128]);
            pixels.extend_from_slice(&[0, 0, 255, 255]);
            let image = SixelImage::try_from_rgba(pixels, 1002, 1).unwrap();
            let options = EncodeOptions {
                max_colors: 2,
                diffusion,
                quantize_method: quantize_method.clone(),
            };
            let encoded = image.encode_with(&options).unwrap();
            let decoded = SixelImage::decode(encoded.as_bytes()).unwrap();

            assert!(decoded.pixels[..4000].as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 0));
            assert_eq!(&decoded.pixels[4000..4004], &[255, 0, 0, 255]);
            assert_eq!(&decoded.pixels[4004..4008], &[0, 0, 255, 255]);
        }
    }
}

#[test]
fn max_colors_clamps_to_2_through_256() {
    // Seven levels per channel give 343 distinct colors, more than the largest palette.
    let pixels: Vec<u8> = (0..343u16)
        .flat_map(|i| [i % 7 * 40, i / 7 % 7 * 40, i / 49 * 40, 255].map(|c| c as u8))
        .collect();
    let image = SixelImage::try_from_rgba(pixels, 343, 1).unwrap();
    let encode = |max_colors| {
        let options = EncodeOptions {
            max_colors,
            diffusion: 0.0,
            ..Default::default()
        };
        image.encode_with(&options).unwrap()
    };
    assert_eq!(encode(2).matches(";2;").count(), 2);
    assert_eq!(encode(0), encode(2));
    assert_eq!(encode(1), encode(2));
    assert_eq!(encode(256).matches(";2;").count(), 256);
    assert_eq!(encode(300), encode(256));
}

#[test]
fn hidden_rgb_does_not_change_encoded_output() {
    let (width, height) = (17, 12);
    let mut first = Vec::new();
    let mut second = Vec::new();
    for y in 0..height {
        for x in 0..width {
            if (x + y) % 3 == 0 {
                first.extend_from_slice(&[0, 255, 0, 0]);
                second.extend_from_slice(&[255, 0, 255, 0]);
            } else {
                let pixel = [(x * 15) as u8, (y * 20) as u8, ((x + y) * 9) as u8, 255];
                first.extend_from_slice(&pixel);
                second.extend_from_slice(&pixel);
            }
        }
    }
    let first = SixelImage::try_from_rgba(first, width, height).unwrap();
    let second = SixelImage::try_from_rgba(second, width, height).unwrap();
    for diffusion in [0.0, 0.5, 1.0] {
        let options = EncodeOptions {
            max_colors: 4,
            diffusion,
            ..Default::default()
        };
        assert_eq!(first.encode_with(&options).unwrap(), second.encode_with(&options).unwrap());
    }
}

#[test]
fn fully_transparent_images_roundtrip() {
    for (width, height) in [(1, 1), (65, 7)] {
        let image = SixelImage::try_from_rgba([255, 0, 255, 0].repeat(width * height), width, height).unwrap();
        let encoded = image.encode().unwrap();
        let decoded = SixelImage::decode(encoded.as_bytes()).unwrap();
        assert_eq!(decoded.dimensions(), (width, height));
        assert!(decoded.pixels.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 0));
    }
}

#[test]
fn a_reused_encoder_matches_a_fresh_encode() {
    // Vary the size, the color count and the transparency, so every reused
    // buffer has to grow, shrink and be cleared between two frames.
    let frames = [
        (4, 3, 255u8, QuantizeMethod::Wu, 0.0),
        (40, 31, 255, QuantizeMethod::Wu, 0.875),
        (7, 9, 100, QuantizeMethod::kmeans(), 0.875),
        (40, 31, 100, QuantizeMethod::Wu, 0.0),
        (1, 1, 255, QuantizeMethod::Wu, 0.875),
    ];
    let mut encoder = SixelEncoder::new();
    let mut out = Vec::new();
    for (width, height, alpha, quantize_method, diffusion) in frames {
        let mut pixels = Vec::with_capacity(width * height * 4);
        for i in 0..width * height {
            pixels.extend_from_slice(&[(i * 7 % 256) as u8, (i * 13 % 256) as u8, (i * 29 % 256) as u8]);
            pixels.push(if i % 3 == 0 { alpha } else { 255 });
        }
        let opts = EncodeOptions {
            diffusion,
            quantize_method,
            ..Default::default()
        };
        encoder = encoder.with_options(opts.clone());
        out.clear();
        encoder.encode_into(&pixels, width, height, &mut out).unwrap();
        assert_eq!(out, icy_sixel::sixel_encode(&pixels, width, height, &opts).unwrap().into_bytes());
    }
}

#[test]
fn encode_into_appends_and_keeps_out_on_an_error() {
    let mut encoder = SixelEncoder::new();
    let mut out = b"prefix".to_vec();
    encoder.encode_into(&[255, 0, 0, 255], 1, 1, &mut out).unwrap();
    let one = out.clone();
    assert!(one.starts_with(b"prefix\x1bP"));

    assert!(matches!(
        encoder.encode_into(&[255, 0, 0], 1, 1, &mut out),
        Err(SixelError::BufferSizeMismatch { expected: 4, actual: 3 })
    ));
    assert_eq!(out, one);

    // A second image appends to the first.
    encoder.encode_into(&[0, 0, 255, 255], 1, 1, &mut out).unwrap();
    assert_eq!(out.iter().filter(|&&b| b == 0x1b).count(), 4);
}

/// One opaque pixel per color, in a single row.
fn row_of(colors: &[[u8; 3]]) -> Vec<u8> {
    colors.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect()
}

/// The colors of a grid of seven levels per channel. No channel percent is 2,
/// so `;2;` counts the palette entries.
fn grid_colors(count: usize) -> Vec<[u8; 3]> {
    (0..343u16)
        .map(|i| [i % 7 * 40, i / 7 % 7 * 40, i / 49 * 40].map(|c| c as u8))
        .take(count)
        .collect()
}

fn encode(encoder: &mut SixelEncoder, rgba: &[u8], width: usize, height: usize) -> String {
    let mut out = Vec::new();
    encoder.encode_into(rgba, width, height, &mut out).unwrap();
    String::from_utf8(out).unwrap()
}

fn exact(max_colors: u16) -> SixelEncoder {
    SixelEncoder::new()
        .with_options(EncodeOptions {
            max_colors,
            diffusion: 0.0,
            ..Default::default()
        })
        .with_exact_palette(true)
}

fn quantized(max_colors: u16) -> SixelEncoder {
    exact(max_colors).with_exact_palette(false)
}

#[test]
fn an_exact_palette_decodes_to_every_color() {
    // Multiples of 51 are whole percents, so the SIXEL keeps them exactly.
    let colors: Vec<[u8; 3]> = (0..21u8).map(|i| [i % 6 * 51, i / 6 * 51, (i * 2) % 6 * 51]).collect();
    let rgba: Vec<u8> = row_of(&colors).repeat(24);
    let encoded = encode(&mut exact(256), &rgba, 21, 24);
    assert_eq!(encoded.matches(";2;").count(), 21);
    assert_eq!(SixelImage::decode(encoded.as_bytes()).unwrap().pixels, rgba);
}

#[test]
fn an_exact_palette_holds_up_to_max_colors() {
    // White goes last, at index 255.
    let mut colors = grid_colors(255);
    colors.push([255, 255, 255]);
    let rgba = row_of(&colors);
    let encoded = encode(&mut exact(256), &rgba, 256, 1);
    assert_eq!(encoded.matches(";2;").count(), 256);
    assert!(encoded.contains("#255;2;100;100;100"));
    let decoded = SixelImage::decode(encoded.as_bytes()).unwrap().pixels;
    assert_eq!(decoded[255 * 4..256 * 4], [255, 255, 255, 255]);
}

#[test]
fn the_quantizer_merges_colors_that_an_exact_palette_keeps() {
    let rgba = row_of(&grid_colors(200));
    let count = |mut encoder| encode(&mut encoder, &rgba, 200, 1).matches(";2;").count();
    assert_eq!(count(exact(256)), 200);
    assert!(count(quantized(256)) < 200, "{}", count(quantized(256)));
}

#[test]
fn more_colors_than_max_colors_go_to_the_quantizer() {
    for (count, max_colors) in [(257, 256), (20, 16)] {
        let rgba = row_of(&grid_colors(count));
        assert_eq!(
            encode(&mut exact(max_colors), &rgba, count, 1),
            encode(&mut quantized(max_colors), &rgba, count, 1)
        );
    }
}

#[test]
fn a_transparent_pixel_takes_no_color_of_an_exact_palette() {
    // Multiples of 51 are whole percents, so the SIXEL keeps them exactly.
    let mut rgba = row_of(&[[0, 0, 0], [51, 102, 153], [255, 204, 0]]);
    rgba.extend(row_of(&grid_colors(297)));
    for pixel in rgba.as_chunks_mut::<4>().0.iter_mut().skip(3) {
        pixel[3] = 0;
    }
    let encoded = encode(&mut exact(256), &rgba, 300, 1);
    assert_eq!(encoded.matches(";2;").count(), 3);
    let decoded = SixelImage::decode(encoded.as_bytes()).unwrap().pixels;
    assert_eq!(decoded[..12], rgba[..12]);
    assert!(decoded[12..].as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 0));
}

#[test]
fn a_transparent_image_is_the_same_with_an_exact_palette() {
    let rgba = [255, 0, 255, 0].repeat(65 * 7);
    assert_eq!(encode(&mut exact(256), &rgba, 65, 7), encode(&mut quantized(256), &rgba, 65, 7));
}

#[test]
fn an_encoder_switches_between_an_exact_palette_and_the_quantizer() {
    let few = row_of(&grid_colors(40)).repeat(3);
    let many = row_of(&grid_colors(300));
    let mut transparent = row_of(&grid_colors(12));
    transparent[3] = 0;
    let mut many_transparent = many.clone();
    many_transparent[3] = 0;
    let frames: [(&[u8], usize, usize); 7] = [
        (&few, 40, 3),
        (&many, 300, 1),
        (&few, 40, 3),
        (&transparent, 12, 1),
        (&many_transparent, 300, 1),
        (&few, 40, 3),
        (&few, 60, 2),
    ];
    let mut encoder = exact(256);
    for (rgba, width, height) in frames {
        assert_eq!(encode(&mut encoder, rgba, width, height), encode(&mut exact(256), rgba, width, height));
    }
}
