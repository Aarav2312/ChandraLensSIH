import { useEffect, useState } from 'react';

/**
 * Load an image URL and return the decoded HTMLImageElement, or null while it
 * is pending. Returns null again if the URL changes before the load settles,
 * so callers never draw a stale frame.
 */
export function useImage(url) {
  const [image, setImage] = useState(null);

  useEffect(() => {
    if (!url) {
      setImage(null);
      return undefined;
    }

    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (!cancelled) setImage(img);
    };
    img.onerror = () => {
      if (!cancelled) setImage(null);
    };
    img.src = url;

    return () => {
      cancelled = true;
    };
  }, [url]);

  return image;
}
