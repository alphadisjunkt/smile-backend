'use strict';

// Inspect raster headers before node-canvas allocates decoded pixel memory.
const MAX_SIDE = 8192;
const MAX_PIXELS = 16 * 1024 * 1024;

function invalid(message, statusCode = 400) {
  return Object.assign(new Error(message), { statusCode });
}

function dimensions(buffer) {
  if (buffer.length >= 24 && buffer.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex')) &&
      buffer.toString('ascii', 12, 16) === 'IHDR') {
    return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
  }
  if (buffer.length >= 4 && buffer[0] === 0xff && buffer[1] === 0xd8) {
    let offset = 2;
    while (offset + 3 < buffer.length) {
      if (buffer[offset++] !== 0xff) return null;
      while (buffer[offset] === 0xff) offset++;
      const marker = buffer[offset++];
      if (marker === 0xda || marker === 0xd9) return null;
      if (marker === 0x01 || marker === 0xd8 || (marker >= 0xd0 && marker <= 0xd7)) continue;
      if (offset + 2 > buffer.length) return null;
      const length = buffer.readUInt16BE(offset);
      if (length < 2 || offset + length > buffer.length) return null;
      const sof = (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker));
      if (sof) {
        if (length < 8) return null;
        return { width: buffer.readUInt16BE(offset + 5), height: buffer.readUInt16BE(offset + 3) };
      }
      offset += length;
    }
  }
  if (buffer.length >= 25 && buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP') {
    const chunk = buffer.toString('ascii', 12, 16);
    if (chunk === 'VP8L' && buffer[20] === 0x2f) {
      const bits = buffer.readUInt32LE(21);
      return { width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
    }
    if (buffer.length >= 30 && chunk === 'VP8X') {
      return { width: buffer.readUIntLE(24, 3) + 1, height: buffer.readUIntLE(27, 3) + 1 };
    }
    if (buffer.length >= 30 && chunk === 'VP8 ' && buffer.subarray(23, 26).equals(Buffer.from('9d012a', 'hex'))) {
      return { width: buffer.readUInt16LE(26) & 0x3fff, height: buffer.readUInt16LE(28) & 0x3fff };
    }
  }
  return null;
}

function decodeRequestImage(image, maxBytes) {
  if (typeof image !== 'string' || !image.trim()) throw invalid('No valid image provided.');
  let encoded = image.trim();
  if (encoded.startsWith('data:')) {
    const prefix = /^data:image\/(?:jpe?g|png|webp);base64,/i.exec(encoded);
    if (!prefix) throw invalid('Unsupported image. Please use a still JPEG, PNG or WebP photo.');
    encoded = encoded.slice(prefix[0].length);
  }
  if (encoded.length > 4 * Math.ceil(maxBytes / 3) + 4) throw invalid('Image too large. Please resize the photo.', 413);
  const base64Data = encoded.replace(/\s/g, '');
  if (!base64Data || base64Data.length % 4 === 1 || !/^[A-Za-z0-9+/]+={0,2}$/.test(base64Data)) {
    throw invalid('Invalid image encoding.');
  }
  const buffer = Buffer.from(base64Data, 'base64');
  if (buffer.length > maxBytes) throw invalid('Image too large. Please resize the photo.', 413);
  const size = dimensions(buffer);
  if (!size || size.width < 32 || size.height < 32) throw invalid('Invalid or unsupported image dimensions.');
  if (size.width > MAX_SIDE || size.height > MAX_SIDE || size.width * size.height > MAX_PIXELS) {
    throw invalid('Image dimensions are too large. Please resize the photo before uploading.', 413);
  }
  // A per-frame pixel budget is insufficient for multi-frame decoders.
  if (buffer.toString('ascii', 0, 4) === 'RIFF' && buffer.toString('ascii', 8, 12) === 'WEBP' &&
      buffer.toString('ascii', 12, 16) === 'VP8X' && (buffer[20] & 2)) {
    throw invalid('Animated images are not supported. Please upload a still photo.');
  }
  if (buffer.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) {
    for (let offset = 8; offset + 12 <= buffer.length;) {
      const length = buffer.readUInt32BE(offset);
      if (length > buffer.length - offset - 12) throw invalid('Invalid PNG chunk.');
      const type = buffer.toString('ascii', offset + 4, offset + 8);
      if (type === 'acTL') throw invalid('Animated images are not supported. Please upload a still photo.');
      offset += length + 12;
    }
  }
  return { base64Data, buffer, ...size };
}

module.exports = { decodeRequestImage, dimensions, MAX_SIDE, MAX_PIXELS };
