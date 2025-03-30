1. Force Crop Scale for InPaint face. This will usually prevent the InPaint from overflowing it's mask. Crops into the masked area, and scales it up. Then scales it back down and merges with the original image latent. The negative effects are mitigated by the diffdiff phase.
2. Merge DiffDiff latent with original image latent before final step. Again to improve the quality. Probably overlap by 0.2 into the blur range so the edge can be refined for one iteration.
3. Add a map_opacity input to control the intensity of the diffdiff phase, but maybe call it something intuitive like blending_intensity.
5. Allow inpaint_ratio as an input. inpaint_steps = int(num_inference_steps * inpaint_ratio)
