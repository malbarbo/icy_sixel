//! Clean-room SIXEL encoder using quantette for high-quality color quantization.
//!
//! This encoder uses the quantette library (MIT/Apache licensed) for optimal
//! color palette generation and dithering, then encodes the result to SIXEL format.

use std::num::NonZeroU32;

use crate::{BackgroundMode, PixelAspectRatio, Result, SixelError, SIXEL_REPEAT_MAX};
use quantette::{
    color_map::{IndexedColorMap, NearestNeighborColorMap},
    color_space::{oklab_to_srgb8, srgb8_to_oklab},
    deps::palette::{Oklab, Srgb},
    dither::FloydSteinberg,
    wu::{BinnerF32x3, WuF32x3},
    ImageRef, PaletteCounts, PaletteSize, Pipeline,
};

// Re-export QuantizeMethod for public API
pub use quantette::QuantizeMethod;

/// Color type for palette entries (RGB).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

/// A compact 1-bit-per-element mask backed by `u64` words.
///
/// Uses 1/8 the memory of a `Vec<bool>`, which improves cache behavior in the
/// per-color band-encoding loop that re-reads the opacity mask many times.
#[derive(Clone, Debug, Default)]
struct BitMask {
    words: Vec<u64>,
}

impl BitMask {
    /// Create a mask with `len` bits, all cleared.
    #[cfg(test)]
    fn zeros(len: usize) -> Self {
        let mut mask = Self::default();
        mask.reset(len);
        mask
    }

    /// Clear every bit and give the mask `len` bits, keeping the allocation.
    fn reset(&mut self, len: usize) {
        self.words.clear();
        self.words.resize(len.div_ceil(64), 0);
    }

    /// Set the bit at `index` to 1.
    #[inline]
    fn set(&mut self, index: usize) {
        self.words[index >> 6] |= 1u64 << (index & 63);
    }

    /// Returns `true` if the first `len` bits are set, `false` otherwise.
    fn is_full(&self, len: usize) -> bool {
        let (words, tail) = (len / 64, len % 64);
        self.words[..words].iter().all(|&word| word == u64::MAX) && (tail == 0 || self.words[words] == (1u64 << tail) - 1)
    }

    /// Return whether the bit at `index` is set.
    #[inline]
    fn get(&self, index: usize) -> bool {
        (self.words[index >> 6] >> (index & 63)) & 1 != 0
    }
}

/// Options for the quantette-based SIXEL encoder.
#[derive(Clone, Debug)]
pub struct EncodeOptions {
    /// Maximum number of colors in the palette (2-256).
    /// Fewer colors = smaller SIXEL output but less accurate colors.
    pub max_colors: u16,

    /// Floyd-Steinberg error diffusion strength (0.0-1.0).
    ///
    /// Controls how much quantization error is spread to neighboring pixels:
    /// - **0.875**: Default (7/8), best for photographs with smooth gradients
    /// - **0.5**: Reduced dithering, less noise, good for graphics
    /// - **0.0**: No dithering, sharp edges but may show color banding
    ///
    /// Higher values produce smoother gradients but may introduce noise.
    /// Lower values preserve sharp edges but may show color banding.
    /// Values are clamped to the range 0.0-1.0.
    pub diffusion: f32,

    /// Color quantization method.
    ///
    /// Available methods:
    /// - [`QuantizeMethod::Wu`]: Wu's color quantizer (default, fast and high quality)
    /// - [`QuantizeMethod::Kmeans`]: K-means clustering (slower but may be more accurate)
    ///
    /// For most use cases, Wu's method provides excellent results.
    pub quantize_method: QuantizeMethod,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            max_colors: 256,
            diffusion: FloydSteinberg::DEFAULT_ERROR_DIFFUSION,
            quantize_method: QuantizeMethod::Wu,
        }
    }
}

/// Encode RGBA image data into a SIXEL string using quantette.
///
/// # Arguments
/// * `rgba` - Raw RGBA pixel data (4 bytes per pixel: R, G, B, A)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `opts` - Encoding options
///
/// # Returns
/// A SIXEL-encoded string that can be displayed on compatible terminals.
/// Pixels with alpha < 128 are transparent and preserved using SIXEL's P2=1 mode.
/// Their RGB values do not affect the palette or error diffusion.
///
/// # Example
/// ```ignore
/// use icy_sixel::{SixelImage, EncodeOptions};
///
/// let rgba = vec![255u8, 0, 0, 255, 0, 255, 0, 255]; // 2 pixels: red, green
/// let image = SixelImage::from_rgba(rgba, 2, 1);
/// let sixel = image.encode_with(&EncodeOptions::default())?;
/// println!("{}", sixel);
/// ```
#[must_use = "this returns the encoded SIXEL string"]
pub fn sixel_encode(rgba: &[u8], width: usize, height: usize, opts: &EncodeOptions) -> Result<String> {
    sixel_encode_impl(rgba, width, height, opts, PixelAspectRatio::default(), BackgroundMode::default())
}

pub(crate) fn sixel_encode_impl(
    rgba: &[u8],
    width: usize,
    height: usize,
    opts: &EncodeOptions,
    pixel_aspect_ratio: PixelAspectRatio,
    background_mode: BackgroundMode,
) -> Result<String> {
    let mut encoder = SixelEncoder::new()
        .with_options(opts.clone())
        .with_aspect_ratio(pixel_aspect_ratio)
        .with_background_mode(background_mode);
    let mut out = Vec::new();
    encoder.encode_into(rgba, width, height, &mut out)?;
    Ok(String::from_utf8(out).expect("the encoder writes ASCII"))
}

/// An encoder that reuses its buffers from one image to the next.
///
/// [`sixel_encode`] and [`SixelImage::encode`](crate::SixelImage::encode)
/// allocate the scratch buffers and the output string on every call. A program
/// that encodes a stream of frames, such as a terminal animation or a video
/// player, keeps one `SixelEncoder` and calls [`encode_into`](Self::encode_into)
/// with a byte buffer it owns, so consecutive frames of the same size allocate
/// nothing in the encoder. The bytes go to the terminal as they are.
///
/// # Example
/// ```rust
/// use std::io::Write;
///
/// use icy_sixel::SixelEncoder;
///
/// let mut encoder = SixelEncoder::new();
/// let mut out = Vec::new();
/// for red in [0u8, 128, 255] {
///     let rgba = [red, 0, 0, 255];
///     out.clear();
///     encoder.encode_into(&rgba, 1, 1, &mut out)?;
///     std::io::stdout().write_all(&out)?;
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct SixelEncoder {
    options: EncodeOptions,
    exact_palette: bool,
    aspect_ratio: PixelAspectRatio,
    background_mode: BackgroundMode,
    scratch: Scratch,
}

impl SixelEncoder {
    /// Create an encoder with the default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the encoding options.
    #[must_use]
    pub fn with_options(mut self, options: EncodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Use the colors of the image as the palette when they fit in
    /// `max_colors`, and quantize only an image with more colors.
    ///
    /// The quantizer merges colors that are close together, so a drawing of
    /// 21 flat colors can come back with 8. An exact palette keeps every
    /// color and can make the output longer, since the SIXEL has one run per
    /// color used in each band. Off by default.
    #[must_use]
    pub fn with_exact_palette(mut self, exact_palette: bool) -> Self {
        self.exact_palette = exact_palette;
        self
    }

    /// Set the pixel aspect ratio the DCS introducer and the raster attributes announce.
    #[must_use]
    pub fn with_aspect_ratio(mut self, aspect_ratio: PixelAspectRatio) -> Self {
        self.aspect_ratio = aspect_ratio;
        self
    }

    /// Set the background mode the DCS introducer announces.
    #[must_use]
    pub fn with_background_mode(mut self, background_mode: BackgroundMode) -> Self {
        self.background_mode = background_mode;
        self
    }

    /// Encode RGBA image data and append the SIXEL to `out`.
    ///
    /// `rgba` holds 4 bytes per pixel and `width` by `height` pixels. A pixel
    /// with alpha below 128 is transparent, as in [`sixel_encode`]. An error
    /// leaves `out` with the contents it had.
    pub fn encode_into(&mut self, rgba: &[u8], width: usize, height: usize, out: &mut Vec<u8>) -> Result<()> {
        crate::validate_encode_dimensions(width, height)?;
        let expected = width.checked_mul(height).and_then(|v| v.checked_mul(4)).ok_or(SixelError::IntegerOverflow)?;
        if rgba.len() != expected {
            return Err(SixelError::BufferSizeMismatch { expected, actual: rgba.len() });
        }
        let image_width = u32::try_from(width).map_err(|_| SixelError::InvalidDimensions { width, height })?;
        let image_height = u32::try_from(height).map_err(|_| SixelError::InvalidDimensions { width, height })?;
        let header = Header {
            width,
            height,
            aspect_ratio: self.aspect_ratio,
            background_mode: self.background_mode,
        };

        let palette_size = PaletteSize::from_u16_clamped(self.options.max_colors.max(2));
        let scratch = &mut self.scratch;
        if self.exact_palette && index_exact_colors(rgba, palette_size.as_usize(), scratch) {
            let Scratch {
                palette,
                indices,
                opacity_mask,
                bands,
                ..
            } = scratch;
            encode_indexed_to_sixel(palette, indices, opacity_mask, header, bands, out);
            return Ok(());
        }

        let diffusion = self.options.diffusion.clamp(0.0, 1.0);
        let wu = matches!(self.options.quantize_method, QuantizeMethod::Wu);
        if diffusion <= 0.0 && wu && count_opaque_colors(rgba, scratch) {
            quantize_counted_colors(scratch, palette_size);
            let Scratch {
                palette,
                indices,
                opacity_mask,
                bands,
                ..
            } = scratch;
            encode_indexed_to_sixel(palette, indices, opacity_mask, header, bands, out);
            return Ok(());
        }

        // Single pass over the RGBA buffer building both the transparency mask
        // (set bit = opaque) and the Srgb<u8> pixels used for quantization
        // (quantette uses palette crate types).
        let pixel_count = expected / 4;
        scratch.opacity_mask.reset(pixel_count);
        scratch.rgb_pixels.clear();
        scratch.rgb_pixels.reserve(pixel_count);
        let mut opaque_count = 0;
        for (i, c) in rgba.as_chunks::<4>().0.iter().enumerate() {
            if c[3] >= 128 {
                scratch.opacity_mask.set(i);
                opaque_count += 1;
            }
            scratch.rgb_pixels.push(Srgb::new(c[0], c[1], c[2]));
        }

        // Use configured quantization method with diffusion-based dithering
        let pipeline = Pipeline::new().palette_size(palette_size).quantize_method(self.options.quantize_method.clone());

        if opaque_count != pixel_count {
            quantize_transparent(scratch, width, pipeline, diffusion)?;
            let Scratch {
                palette,
                indices,
                opacity_mask,
                bands,
                ..
            } = scratch;
            encode_indexed_to_sixel(palette, indices, opacity_mask, header, bands, out);
            return Ok(());
        }

        // Create image reference for quantette
        let image = ImageRef::new(image_width, image_height, &scratch.rgb_pixels).map_err(|e| SixelError::Quantization(e.to_string()))?;

        // Apply dithering based on diffusion setting
        let indexed_image = if diffusion <= 0.0 {
            // No dithering - sharp edges, may show banding. Quantette dedups
            // the pixels by itself only for a large image. A drawing repeats
            // few colors, so the dedup costs less than the conversion of every
            // pixel to Oklab that it saves.
            pipeline.ditherer(None).dedup(true).input_image(image).output_srgb8_indexed_image()
        } else {
            // Use Floyd-Steinberg dithering with specified diffusion strength
            let ditherer = FloydSteinberg::with_error_diffusion(diffusion).unwrap_or_default();
            pipeline.ditherer(ditherer).input_image(image).output_srgb8_indexed_image()
        };

        // Extract the palette. The indices stay in the image quantette returned,
        // so they are read in place.
        scratch.palette.clear();
        scratch.palette.extend(indexed_image.palette().iter().map(|c| Rgb {
            r: c.red,
            g: c.green,
            b: c.blue,
        }));

        // Encode to SIXEL with transparency support
        let Scratch {
            palette, opacity_mask, bands, ..
        } = scratch;
        encode_indexed_to_sixel(palette, indexed_image.indices(), opacity_mask, header, bands, out);
        Ok(())
    }
}

/// The buffers an encoder keeps from one image to the next.
#[derive(Clone, Debug, Default)]
struct Scratch {
    /// The RGB of every pixel, as quantette takes them.
    rgb_pixels: Vec<Srgb<u8>>,
    /// One bit per pixel, set when the pixel is opaque.
    opacity_mask: BitMask,
    /// The quantized palette.
    palette: Vec<Rgb>,
    /// One palette index per pixel, filled when the image has transparency or
    /// the encoder takes the exact palette. An opaque image that goes through quantette reads the
    /// indices of the quantette image in place.
    indices: Vec<u8>,
    /// The buffers of [`index_exact_colors`] and [`count_opaque_colors`].
    counted: Counted,
    /// The buffers of the band loop.
    bands: Bands,
    /// The buffers of the masked dithering.
    dither: Dither,
}

/// What the DCS introducer and the raster attributes announce about the image.
#[derive(Clone, Copy, Debug)]
struct Header {
    width: usize,
    height: usize,
    aspect_ratio: PixelAspectRatio,
    background_mode: BackgroundMode,
}

/// The buffers of the band loop in [`encode_indexed_to_sixel`].
#[derive(Clone, Debug, Default)]
struct Bands {
    /// The 6-bit sixel value of every (color, column) pair in the band.
    sixels: Vec<u8>,
    /// The first and one past the last column of the color in the band, or
    /// `None` when the color does not appear in it.
    spans: Vec<Option<(usize, usize)>>,
}

/// The distinct colors of an image, which [`index_exact_colors`] and
/// [`count_opaque_colors`] fill.
#[derive(Clone, Debug, Default)]
struct Counted {
    /// A hash table. Each slot holds one more than the index of a color in
    /// `colors`, or `None`. The length is a power of two, and at most half of
    /// the slots hold a color.
    slots: Vec<Option<NonZeroU32>>,
    /// The distinct colors, in the order of their first pixel.
    colors: Vec<Srgb<u8>>,
    /// The number of pixels of each color in `colors`.
    counts: Vec<u32>,
    /// The index in `colors` of the color of every pixel. Only
    /// [`count_opaque_colors`] fills it and `counts`.
    pixels: Vec<u32>,
}

/// The buffers of [`map_visible_pixels`].
#[derive(Clone, Debug, Default)]
struct Dither {
    /// The opaque pixels alone, which train the palette.
    opaque_pixels: Vec<Srgb<u8>>,
    /// Every pixel in Oklab, where the error diffusion runs.
    colors: Vec<Oklab>,
    /// The error carried into the row in progress and into the row below it.
    current: Vec<[f32; 3]>,
    next: Vec<[f32; 3]>,
}

/// Returns `true` if the colors of the opaque pixels fit in `budget`, `false`
/// otherwise. On `true`, `scratch.palette` holds those colors,
/// `scratch.indices` the index of the color of every opaque pixel and
/// `scratch.opacity_mask` the opaque pixels. On `false`, the three buffers hold
/// a part of the image.
fn index_exact_colors(rgba: &[u8], budget: usize, scratch: &mut Scratch) -> bool {
    let Scratch {
        opacity_mask,
        palette,
        indices,
        counted,
        ..
    } = scratch;
    let pixels = rgba.as_chunks::<4>().0;
    opacity_mask.reset(pixels.len());
    palette.clear();
    indices.clear();
    indices.resize(pixels.len(), 0);
    counted.clear();
    // A run of pixels of one color skips the table.
    let mut last: Option<([u8; 3], u8)> = None;
    for (i, c) in pixels.iter().enumerate() {
        if c[3] < 128 {
            continue;
        }
        opacity_mask.set(i);
        let color = [c[0], c[1], c[2]];
        let index = match last {
            Some((last_color, index)) if last_color == color => index,
            _ => {
                let index = counted.find_or_insert(color);
                if counted.colors.len() > budget {
                    return false;
                }
                u8::try_from(index).expect("the budget is at most 256")
            }
        };
        last = Some((color, index));
        indices[i] = index;
    }
    palette.extend(counted.colors.iter().map(|c| Rgb {
        r: c.red,
        g: c.green,
        b: c.blue,
    }));
    true
}

/// The number of slots of the table of [`Counted`] at the start of every
/// image. The table grows for an image of more colors.
const MIN_COUNTED_SLOTS: usize = 4096;

/// Returns `true` if every pixel is opaque, `false` otherwise. On `true`,
/// `scratch.counted` holds the distinct colors of the image, the number of
/// pixels of each one and the color of every pixel, and `scratch.opacity_mask`
/// has every bit set.
///
/// Quantette finds the distinct colors with a radix sort of a copy of the
/// pixels. A hash table finds them in one pass over `rgba`, and a run of
/// pixels of one color skips the table. A transparent pixel stops the pass,
/// so an image with one near the end costs a pass for nothing.
fn count_opaque_colors(rgba: &[u8], scratch: &mut Scratch) -> bool {
    let Scratch { opacity_mask, counted, .. } = scratch;
    let pixels = rgba.as_chunks::<4>().0;
    opacity_mask.reset(pixels.len());
    counted.clear();
    counted.pixels.reserve(pixels.len());
    let mut last: Option<([u8; 3], u32)> = None;
    for (i, c) in pixels.iter().enumerate() {
        if c[3] < 128 {
            return false;
        }
        opacity_mask.set(i);
        let color = [c[0], c[1], c[2]];
        let index = match last {
            Some((last_color, index)) if last_color == color => index,
            _ => counted.find_or_insert(color),
        };
        last = Some((color, index));
        counted.counts[index as usize] += 1;
        counted.pixels.push(index);
    }
    true
}

impl Counted {
    /// Empty the table and the colors.
    fn clear(&mut self) {
        self.slots.clear();
        self.slots.resize(MIN_COUNTED_SLOTS, None);
        self.colors.clear();
        self.counts.clear();
        self.pixels.clear();
    }

    /// Returns the index of `color` in `colors`, which joins `colors` with a
    /// count of zero if it is new.
    fn find_or_insert(&mut self, [r, g, b]: [u8; 3]) -> u32 {
        let color = Srgb::new(r, g, b);
        let mut slot = self.slot_of(color);
        loop {
            match self.slots[slot] {
                None => break,
                Some(slot) if self.colors[slot.get() as usize - 1] == color => return slot.get() - 1,
                Some(_) => slot = (slot + 1) & (self.slots.len() - 1),
            }
        }
        let index = u32::try_from(self.colors.len()).expect("an image has at most 2^26 pixels");
        self.slots[slot] = NonZeroU32::new(index + 1);
        self.colors.push(color);
        self.counts.push(0);
        if self.colors.len() * 2 > self.slots.len() {
            self.grow();
        }
        index
    }

    /// Double the slots and put every color back.
    fn grow(&mut self) {
        let len = self.slots.len() * 2;
        self.slots.clear();
        self.slots.resize(len, None);
        let mask = self.slots.len() - 1;
        for (index, &color) in (1..).zip(&self.colors) {
            let mut slot = self.slot_of(color);
            while self.slots[slot].is_some() {
                slot = (slot + 1) & mask;
            }
            self.slots[slot] = NonZeroU32::new(index);
        }
    }

    /// The first slot to try for `color`.
    fn slot_of(&self, color: Srgb<u8>) -> usize {
        fibonacci_slot([color.red, color.green, color.blue], self.slots.len())
    }
}

/// The first slot to try for `rgb` in a hash table of `len` slots, a power of
/// two. Fibonacci hashing takes the top bits of the product, which mix every
/// channel.
fn fibonacci_slot(rgb: [u8; 3], len: usize) -> usize {
    let [r, g, b] = rgb;
    let key = u32::from_be_bytes([0, r, g, b]);
    (key.wrapping_mul(0x9E37_79B1) >> (32 - len.trailing_zeros())) as usize
}

/// Quantize the colors of `scratch.counted` with Wu, and leave the result in
/// `scratch.palette` and `scratch.indices`. Wu weighs each distinct color by
/// its count, so the palette is the one of Wu over every pixel, up to the
/// order of the float sums of its histogram.
fn quantize_counted_colors(scratch: &mut Scratch, palette_size: PaletteSize) {
    let Scratch { palette, indices, counted, .. } = scratch;
    let oklab = srgb8_to_oklab(&counted.colors);
    let palette_counts = PaletteCounts::new(oklab, counted.counts.clone()).expect("an image has at most 2^26 pixels");
    let color_map = WuF32x3::run_palette_counts(&palette_counts, BinnerF32x3::oklab_from_srgb8())
        .expect("an image has a pixel")
        .color_map(palette_size);
    let palette_indices = color_map.map_to_indices(palette_counts.palette());
    indices.clear();
    indices.extend(counted.pixels.iter().map(|&color| palette_indices[color as usize]));
    palette.clear();
    palette.extend(oklab_to_srgb8(color_map.palette()).into_iter().map(|c| Rgb {
        r: c.red,
        g: c.green,
        b: c.blue,
    }));
}

/// Quantette has no alpha-mask support. Train its palette only on visible pixels,
/// then map the original layout without diffusing error through transparent pixels.
/// Leaves the result in `scratch.palette` and `scratch.indices`.
fn quantize_transparent(scratch: &mut Scratch, width: usize, pipeline: Pipeline, diffusion: f32) -> Result<()> {
    let Scratch {
        rgb_pixels,
        opacity_mask,
        palette,
        indices,
        dither,
        ..
    } = scratch;
    palette.clear();
    indices.clear();
    dither.opaque_pixels.clear();
    dither
        .opaque_pixels
        .extend(rgb_pixels.iter().enumerate().filter_map(|(i, &pixel)| opacity_mask.get(i).then_some(pixel)));
    if dither.opaque_pixels.is_empty() {
        // Raster attributes preserve the dimensions; no color or index is used.
        return Ok(());
    }

    let quantized = pipeline
        .input_slice(&dither.opaque_pixels)
        .map_err(|e| SixelError::Quantization(e.to_string()))?
        .output_oklab_palette();
    let color_map = NearestNeighborColorMap::new(quantized);
    dither.colors.clear();
    dither.colors.extend(srgb8_to_oklab(rgb_pixels));
    indices.resize(dither.colors.len(), 0);
    map_visible_pixels(dither, opacity_mask, width, &color_map, diffusion, indices);
    palette.extend(oklab_to_srgb8(color_map.palette()).into_iter().map(|c| Rgb {
        r: c.red,
        g: c.green,
        b: c.blue,
    }));
    Ok(())
}

/// Serpentine Floyd–Steinberg in Oklab, with transparent pixels acting as sinks
/// for incoming error. The palette and nearest-neighbor search come from quantette.
/// `indices` holds one zeroed entry per pixel and receives the palette index of
/// every opaque pixel.
fn map_visible_pixels(
    dither: &mut Dither,
    opacity_mask: &BitMask,
    width: usize,
    color_map: &NearestNeighborColorMap<Oklab, f32, 3>,
    diffusion: f32,
    indices: &mut [u8],
) {
    let colors = &dither.colors;
    if diffusion <= 0.0 {
        for (i, color) in colors.iter().enumerate() {
            if opacity_mask.get(i) {
                indices[i] = color_map.palette_index(color);
            }
        }
        return;
    }

    // Match the opaque pipeline's fallback for non-finite diffusion values.
    let diffusion = FloydSteinberg::with_error_diffusion(diffusion).unwrap_or_default().error_diffusion();
    let (current, next) = (&mut dither.current, &mut dither.next);
    current.clear();
    current.resize(width + 2, [0.0; 3]);
    next.clear();
    next.resize(width + 2, [0.0; 3]);
    for (y, row) in colors.chunks_exact(width).enumerate() {
        let left_to_right = y % 2 == 0;
        for step in 0..width {
            let x = if left_to_right { step } else { width - 1 - step };
            let i = y * width + x;
            if !opacity_mask.get(i) {
                continue;
            }
            let color = row[x];
            let adjusted = Oklab::new(color.l + current[x + 1][0], color.a + current[x + 1][1], color.b + current[x + 1][2]);
            let index = color_map.palette_index(&adjusted);
            indices[i] = index;
            let nearest = color_map.palette()[index];
            let error = [adjusted.l - nearest.l, adjusted.a - nearest.a, adjusted.b - nearest.b];
            let (forward, backward) = if left_to_right { (x + 2, x) } else { (x, x + 2) };
            for (channel, error) in error.into_iter().enumerate() {
                let error = error * diffusion / 16.0;
                current[forward][channel] += error * 7.0;
                next[backward][channel] += error * 3.0;
                next[x + 1][channel] += error * 5.0;
                next[forward][channel] += error;
            }
        }
        std::mem::swap(current, next);
        next.fill([0.0; 3]);
    }
}

/// Encode RGBA with default options.
#[inline]
#[deprecated(since = "0.5.0", note = "use SixelImage::from_rgba().encode() instead")]
#[must_use = "this returns the encoded SIXEL string"]
pub fn sixel_encode_default(rgba: &[u8], width: usize, height: usize) -> Result<String> {
    #[allow(deprecated)]
    sixel_encode(rgba, width, height, &EncodeOptions::default())
}

fn encode_indexed_to_sixel(palette: &[Rgb], indices: &[u8], opacity_mask: &BitMask, header: Header, bands: &mut Bands, out: &mut Vec<u8>) {
    let Header {
        width,
        height,
        aspect_ratio,
        background_mode,
    } = header;
    // DCS introducer for SIXEL: ESC P p1 ; p2 ; p3 q
    // p1=aspect ratio, p2=background mode, p3=0 (grid size default)
    out.extend_from_slice(b"\x1bP");
    write_number(out, aspect_ratio.to_p1_value() as usize);
    out.push(b';');
    write_number(out, background_mode.to_p2_value() as usize);
    out.extend_from_slice(b";0q");

    // Set raster attributes: " Pan ; Pad ; Ph ; Pv
    // Pan:Pad is vertical:horizontal, so it mirrors the P1 macro parameter.
    // Emitting this is required for terminals and multiplexers (e.g. tmux) that
    // drop or rewrite P1 but forward the raster attributes.
    out.push(b'"');
    write_number(out, aspect_ratio.pad() as usize);
    out.push(b';');
    write_number(out, aspect_ratio.pan() as usize);
    out.push(b';');
    write_number(out, width);
    out.push(b';');
    write_number(out, height);

    // Define palette in RGB percent (0-100)
    for (i, c) in palette.iter().enumerate() {
        // Round to the nearest percentage instead of introducing a dark bias.
        let r = (c.r as u32 * 100 + 127) / 255;
        let g = (c.g as u32 * 100 + 127) / 255;
        let b = (c.b as u32 * 100 + 127) / 255;
        out.push(b'#');
        write_number(out, i);
        out.push(b';');
        out.push(b'2');
        out.push(b';');
        write_number(out, r as usize);
        out.push(b';');
        write_number(out, g as usize);
        out.push(b';');
        write_number(out, b as usize);
    }

    let band_count = height.div_ceil(6);
    let palette_len = palette.len();

    // Scratch buffer holding the 6-bit sixel value for every (color, column)
    // pair in the current band. Reused across bands; only the rows of colors
    // actually used in a band are cleared, so this stays cheap. The product
    // fits, since the palette holds at most 256 colors and the width is at most
    // SIXEL_WIDTH_LIMIT.
    let scratch_len = palette_len * width;
    let Bands { sixels, spans } = bands;
    sixels.clear();
    sixels.resize(scratch_len, 0);
    spans.clear();
    spans.resize(palette_len, None);

    let opacity = (!opacity_mask.is_full(width * height)).then_some(opacity_mask);
    for band in 0..band_count {
        let y0 = band * 6;
        let y_max = usize::min(y0 + 6, height);

        // Reset only the rows touched by the previous band.
        for (color_index, span) in spans.iter_mut().enumerate() {
            if let Some((start, end)) = span.take() {
                sixels[color_index * width + start..color_index * width + end].fill(0);
            }
        }

        // Single pass over the band: scatter the bit of each opaque pixel
        // into the scratch buffer keyed by its color. A drawing has long
        // runs of one color, so the pass goes run by run.
        for y in y0..y_max {
            let bit = 1u8 << (y - y0);
            let first = y * width;
            let mut x = 0;
            while x < width {
                let (end, index) = run_at(indices, opacity, first, x, width);
                if let Some(index) = index {
                    let color_index = usize::from(index);
                    for sixel in &mut sixels[color_index * width + x..color_index * width + end] {
                        *sixel |= bit;
                    }
                    let span = &mut spans[color_index];
                    *span = Some(match *span {
                        Some((start, span_end)) => (start.min(x), span_end.max(end)),
                        None => (x, end),
                    });
                }
                x = end;
            }
        }

        // Emit each used color, run-length encoding consecutive identical sixels.
        for (color_index, span) in spans.iter().enumerate() {
            let Some((start, end)) = *span else {
                continue;
            };

            // Select color map register
            out.push(b'#');
            write_number(out, color_index);

            // The columns before the span are empty, and the ones after it
            // need nothing, since the next color starts again at column 0.
            write_empty_run(out, start);
            let row = &sixels[color_index * width..color_index * width + end];
            let mut x = start;
            while x < end {
                let bits = row[x];

                // Run-length encode consecutive identical sixel values
                let mut run_len = 1usize;
                while run_len < SIXEL_REPEAT_MAX && x + run_len < end && row[x + run_len] == bits {
                    run_len += 1;
                }

                // Write RLE or raw sixels
                if run_len > 3 {
                    out.push(b'!');
                    write_number(out, run_len);
                    out.push(63 + bits);
                } else {
                    let ch = 63 + bits;
                    for _ in 0..run_len {
                        out.push(ch);
                    }
                }
                x += run_len;
            }

            // Carriage return to start of band for next color overlay
            out.push(b'$');
        }

        // Move to next band
        out.push(b'-');
    }

    // String terminator: ESC \
    out.push(b'\x1b');
    out.push(b'\\');
}

/// The run that starts at column `x` of the row that starts at pixel
/// `first`, as the column after it and its index. A run holds opaque pixels
/// of one index, or transparent pixels, which have `None` for an index and
/// may have no entry in `indices`. `opacity` is `None` when every pixel is
/// opaque, which spares the mask lookup.
fn run_at(indices: &[u8], opacity: Option<&BitMask>, first: usize, x: usize, width: usize) -> (usize, Option<u8>) {
    let mut end = x + 1;
    match opacity {
        Some(mask) if !mask.get(first + x) => {
            while end < width && !mask.get(first + end) {
                end += 1;
            }
            (end, None)
        }
        Some(mask) => {
            let index = indices[first + x];
            while end < width && mask.get(first + end) && indices[first + end] == index {
                end += 1;
            }
            (end, Some(index))
        }
        None => {
            let index = indices[first + x];
            while end < width && indices[first + end] == index {
                end += 1;
            }
            (end, Some(index))
        }
    }
}

/// Write `len` empty sixels.
fn write_empty_run(out: &mut Vec<u8>, mut len: usize) {
    while len > 0 {
        let run = len.min(SIXEL_REPEAT_MAX);
        if run > 3 {
            out.push(b'!');
            write_number(out, run);
            out.push(b'?');
        } else {
            out.extend(std::iter::repeat_n(b'?', run));
        }
        len -= run;
    }
}

/// Fast number to string without allocation
#[inline]
fn write_number(out: &mut Vec<u8>, mut n: usize) {
    if n == 0 {
        out.push(b'0');
        return;
    }

    let mut buf = [0u8; 20];
    let mut i = buf.len();

    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }

    out.extend_from_slice(&buf[i..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opaque(colors: &[[u8; 3]]) -> Vec<u8> {
        colors.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect()
    }

    /// 128 by 64 pixels of 8192 colors, one per pixel.
    fn gradient() -> Vec<[u8; 3]> {
        (0..128 * 64).map(|i| [(i % 128) as u8 * 2, (i / 128) as u8 * 4, (i * 7) as u8]).collect()
    }

    fn encode(encoder: &mut SixelEncoder, colors: &[[u8; 3]]) -> String {
        let mut out = Vec::new();
        encoder.encode_into(&opaque(colors), 128, 64, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn counted_colors_quantize_as_the_pipeline_does() {
        // 5038 colors of 1 to over 8 pixels each, which grow the table twice.
        let (width, height) = (128, 128);
        let gradient = gradient();
        let colors: Vec<[u8; 3]> = (0..width * height).map(|i| gradient[i * i / 7 % 8192]).collect();
        let size = PaletteSize::from_u16_clamped(256);
        let pixels: Vec<Srgb<u8>> = colors.iter().map(|&[r, g, b]| Srgb::new(r, g, b)).collect();
        let image = ImageRef::new(width as u32, height as u32, &pixels).unwrap();
        let expected = Pipeline::new()
            .palette_size(size)
            .ditherer(None)
            .dedup(true)
            .input_image(image)
            .output_srgb8_indexed_image();
        let expected: Vec<Srgb<u8>> = expected.indices().iter().map(|&i| expected.palette()[usize::from(i)]).collect();

        let mut scratch = Scratch::default();
        assert!(count_opaque_colors(&opaque(&colors), &mut scratch));
        quantize_counted_colors(&mut scratch, size);
        let got: Vec<Srgb<u8>> = scratch
            .indices
            .iter()
            .map(|&i| {
                let c = scratch.palette[usize::from(i)];
                Srgb::new(c.r, c.g, c.b)
            })
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn counted_colors_hold_each_color_once_with_its_pixels() {
        let (a, b) = ([1, 2, 3], [4, 5, 6]);
        let mut scratch = Scratch::default();
        assert!(count_opaque_colors(&opaque(&[a, a, b, a]), &mut scratch));
        let counted = &scratch.counted;
        assert_eq!(counted.colors, [Srgb::new(1, 2, 3), Srgb::new(4, 5, 6)]);
        assert_eq!(counted.counts, [3, 1]);
        assert_eq!(counted.pixels, [0, 0, 1, 0]);
        assert!((0..4).all(|i| scratch.opacity_mask.get(i)));
    }

    #[test]
    fn a_reused_table_counts_the_next_image_alone() {
        let (a, b) = ([1, 2, 3], [4, 5, 6]);
        let mut scratch = Scratch::default();
        let mut first = gradient();
        first.push(a);
        assert!(count_opaque_colors(&opaque(&first), &mut scratch));
        assert!(count_opaque_colors(&opaque(&[b, b]), &mut scratch));
        assert_eq!(scratch.counted.colors, [Srgb::new(4, 5, 6)]);
        assert_eq!(scratch.counted.counts, [2]);
        assert_eq!(scratch.counted.pixels, [0, 0]);
    }

    #[test]
    fn dithering_goes_on_past_256_colors() {
        let colors = gradient();
        let with = |diffusion| {
            let options = EncodeOptions {
                diffusion,
                ..Default::default()
            };
            encode(&mut SixelEncoder::new().with_options(options), &colors)
        };
        assert_ne!(with(0.0), with(0.875));
    }

    #[test]
    fn a_custom_palette_without_dithering_keeps_its_colors() {
        let black_and_white = quantette::PaletteBuf::new(vec![Srgb::new(0, 0, 0), Srgb::new(255, 255, 255)]).unwrap();
        let options = EncodeOptions {
            diffusion: 0.0,
            quantize_method: black_and_white.into(),
            ..Default::default()
        };
        let encoded = encode(&mut SixelEncoder::new().with_options(options), &gradient());
        assert_eq!(encoded.matches(";2;").count(), 2, "{encoded}");
    }

    #[test]
    fn a_mask_is_full_when_its_first_bits_are_set() {
        for len in [1, 63, 64, 65, 128] {
            let mut mask = BitMask::zeros(len);
            (0..len).for_each(|i| mask.set(i));
            assert!(mask.is_full(len), "{len}");
            for hole in [0, len / 2, len - 1] {
                let mut mask = BitMask::zeros(len);
                (0..len).filter(|&i| i != hole).for_each(|i| mask.set(i));
                assert!(!mask.is_full(len), "{len} {hole}");
            }
        }
    }

    #[test]
    fn a_transparent_pixel_stops_the_count() {
        let mut rgba = opaque(&[[1, 2, 3], [4, 5, 6]]);
        rgba[7] = 0;
        assert!(!count_opaque_colors(&rgba, &mut Scratch::default()));
    }

    /// Run [`map_visible_pixels`] over `colors` and return the indices.
    fn map(colors: &[Oklab], mask: &BitMask, width: usize, color_map: &NearestNeighborColorMap<Oklab, f32, 3>, diffusion: f32) -> Vec<u8> {
        let mut dither = Dither {
            colors: colors.to_vec(),
            ..Default::default()
        };
        let mut indices = vec![0; colors.len()];
        map_visible_pixels(&mut dither, mask, width, color_map, diffusion, &mut indices);
        indices
    }

    #[test]
    fn transparent_pixels_stop_error_diffusion() {
        let palette = quantette::PaletteBuf::new(vec![Oklab::new(0.0, 0.0, 0.0), Oklab::new(1.0, 0.0, 0.0)]).unwrap();
        let color_map = NearestNeighborColorMap::new(palette);
        // Horizontal, vertical and reverse-scan barriers, including an empty row.
        for (width, len, visible) in [(3, 3, [0, 2]), (1, 3, [0, 2]), (3, 6, [3, 5]), (3, 9, [1, 7])] {
            let colors = vec![Oklab::new(0.49, 0.0, 0.0); len];
            let mut mask = BitMask::zeros(len);
            for i in visible {
                mask.set(i);
            }
            let indices = map(&colors, &mask, width, &color_map, 1.0);
            for i in visible {
                assert_eq!(indices[i], 0, "error must not cross transparent pixels");
            }
        }
    }

    #[test]
    fn masked_dithering_changes_visible_gradients() {
        let palette = quantette::PaletteBuf::new(vec![Oklab::new(0.0, 0.0, 0.0), Oklab::new(1.0, 0.0, 0.0)]).unwrap();
        let color_map = NearestNeighborColorMap::new(palette);
        let colors = vec![Oklab::new(0.49, 0.0, 0.0); 3];
        let mut mask = BitMask::zeros(3);
        mask.set(0);
        mask.set(1);
        assert_eq!(map(&colors, &mask, 3, &color_map, 0.0), [0, 0, 0]);
        assert_eq!(map(&colors, &mask, 3, &color_map, 1.0), [0, 1, 0]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_encode_simple() {
        let rgba = vec![255u8, 0, 0, 255]; // 1x1 red pixel
        let result = sixel_encode(&rgba, 1, 1, &EncodeOptions::default());
        assert!(result.is_ok());
        let sixel = result.unwrap();
        assert!(sixel.starts_with("\x1bP9;1;0q\"1;1;1;1"));
        assert!(sixel.ends_with("\x1b\\"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_encode_2x2() {
        let rgba = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 0, 255, // yellow
        ];
        let result = sixel_encode(&rgba, 2, 2, &EncodeOptions::default());
        assert!(result.is_ok());
    }

    //This test is a compatibility test, the P1 value needs to be 7-9 in order to have square pixels. If it is set to a different value
    //like 0 (default) then most terminals will do things correctly but the Windows Terminal will default to a non-square sixel making the
    //image print out with an incorrect aspect ratio.
    #[test]
    #[allow(deprecated)]
    fn test_encode_is_set_to_square_pixels() {
        let rgba = vec![
            255, 0, 0, 255, // red
            0, 0, 255, 255, // blue
        ];
        let sixel = sixel_encode(&rgba, 2, 1, &EncodeOptions::default()).unwrap();
        assert!(sixel.contains("\x1bP9;"));
        // Raster attributes must also state a 1:1 ratio: multiplexers such as tmux
        // rewrite P1 but pass the raster attributes through unchanged.
        assert!(sixel.contains("\"1;1;2;1"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_invalid_dimensions() {
        let rgba = vec![0u8; 16];

        assert!(sixel_encode(&rgba, 0, 4, &EncodeOptions::default()).is_err());
        assert!(sixel_encode(&rgba, 4, 0, &EncodeOptions::default()).is_err());
        assert!(sixel_encode(&rgba, 10, 10, &EncodeOptions::default()).is_err());
    }
}
