import cv2
import numpy as np


class ImageEnhancer:
    @staticmethod
    def _validate(image):
        if (
            image is None
            or not isinstance(image, np.ndarray)
            or image.size == 0
        ):
            raise ValueError("A non-empty image is required.")

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB or grayscale image.")

        return np.clip(image, 0, 255).astype(np.uint8).copy()

    def metrics(self, image):
        image = self._validate(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return {
            "brightness": float(np.mean(gray)),
            "contrast": float(np.std(gray)),
            "sharpness": float(
                cv2.Laplacian(gray, cv2.CV_64F).var()
            ),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        }

    def enhance_low_light(
        self,
        image,
        clip_limit=2.5,
        tile_grid_size=(8, 8),
    ):
        image = self._validate(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lightness, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tile_grid_size,
        )
        enhanced_l = clahe.apply(lightness)

        return cv2.cvtColor(
            cv2.merge((enhanced_l, a, b)),
            cv2.COLOR_LAB2RGB,
        )

    def gamma_correction(self, image, gamma=1.4):
        image = self._validate(image)
        gamma = max(0.1, float(gamma))

        lut = np.array(
            [
                min(
                    255,
                    round(((i / 255.0) ** (1.0 / gamma)) * 255),
                )
                for i in range(256)
            ],
            dtype=np.uint8,
        )

        return cv2.LUT(image, lut)

    def denoise(self, image):
        image = self._validate(image)

        return cv2.fastNlMeansDenoisingColored(
            image, None, 5, 5, 7, 21
        )

    def sharpen(self, image, amount=0.5):
        image = self._validate(image)
        amount = float(np.clip(amount, 0.0, 3.0))

        blurred = cv2.GaussianBlur(image, (0, 0), 2.0)
        result = cv2.addWeighted(
            image,
            1.0 + amount,
            blurred,
            -amount,
            0,
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def night_vision(self, image):
        image = self._validate(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        enhanced = self.enhance_low_light(gray_rgb)
        gray = enhanced[:, :, 0]

        green = gray.astype(np.float32)
        red = green * 0.12
        blue = green * 0.18

        output = np.stack((red, green, blue), axis=-1)

        height, width = output.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        cx, cy = width / 2.0, height / 2.0

        radius = np.sqrt(
            ((xx - cx) / max(cx, 1)) ** 2
            + ((yy - cy) / max(cy, 1)) ** 2
        )
        vignette = np.clip(1.0 - 0.38 * radius**2, 0.45, 1.0)

        return np.clip(
            output * vignette[..., None],
            0,
            255,
        ).astype(np.uint8)

    def prepare(
        self,
        image,
        enhance=True,
        night_vision=False,
        adaptive=True,
    ):
        original = self._validate(image)
        result = original.copy()
        applied = []
        before = self.metrics(original)

        if enhance:
            result = self.enhance_low_light(result)
            applied.append("CLAHE")

            if adaptive and before["brightness"] < 65:
                result = self.gamma_correction(
                    result, gamma=1.6
                )
                applied.append("gamma_correction")

            if adaptive and before["sharpness"] < 40:
                result = self.sharpen(result, amount=0.35)
                applied.append("sharpen")

        if night_vision:
            result = self.night_vision(result)
            applied.append("night_vision")

        return result, {
            "applied": applied,
            "enhancement_applied": bool(
                enhance and any(
                    step in applied
                    for step in (
                        "CLAHE",
                        "gamma_correction",
                        "sharpen",
                    )
                )
            ),
            "night_vision_applied": bool(night_vision),
            "quality_before": before,
            "quality_after": self.metrics(result),
        }