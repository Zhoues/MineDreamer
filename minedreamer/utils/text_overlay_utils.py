import numpy as np
import cv2

from minedreamer.config import FONT, THICKNESS, TEXT_POSITION, FONT_SCALE, LIGHT_ORANGE_COLOR, DARK_ORANGE_COLOR


def put_colored_text(frame, text, position, font, font_scale, thickness=THICKNESS, color=LIGHT_ORANGE_COLOR):
    return cv2.putText(frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def create_video_frame_w_prompt_on_right_side(gameplay_pov, prompt, step=None, location_stats=None):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    prompt_section = created_fitted_text_image(frame.shape[1] // 2, prompt,
                                               background_color=(0, 0, 0),
                                               text_color=(255, 255, 255))
    pad_top_height = (frame.shape[0] - prompt_section.shape[0]) // 2
    pad_top = np.zeros((pad_top_height, prompt_section.shape[1], 3), dtype=np.uint8)
    pad_bottom_height = frame.shape[0] - pad_top_height - prompt_section.shape[0]
    pad_bottom = np.zeros((pad_bottom_height, prompt_section.shape[1], 3), dtype=np.uint8)
    prompt_section = np.vstack((pad_top, prompt_section, pad_bottom))
    frame = np.hstack((frame, prompt_section))

    if step is None or location_stats is None:
        return frame

    text_position = TEXT_POSITION

    info_text_lines = [
        f"Step: {step}",
        f"Pos: {[abs(int(location_stats['xpos'])) , abs(int(location_stats['ypos'])), abs(int(location_stats['zpos']))]}"
    ]

    for line in info_text_lines:
        put_colored_text(frame, line, text_position, FONT, FONT_SCALE, THICKNESS, LIGHT_ORANGE_COLOR)
        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)[0]
        text_position = (text_position[0], text_position[1] + text_size[1] + 10)

    return frame

def create_video_frame_w_prompt_on_frame(gameplay_pov, prompt, step=None, location_stats=None):
    """Creates a frame for the generated video with the gameplay POV and the prompt text on the right side."""
    frame = cv2.cvtColor(gameplay_pov, cv2.COLOR_RGB2BGR)
    
    text_position = TEXT_POSITION

    prompt_text_line = [
        f"Prompt: {prompt}"
    ]

    for line in prompt_text_line:
        put_colored_text(frame, line, text_position, FONT, FONT_SCALE, THICKNESS, DARK_ORANGE_COLOR)
        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)[0]
        text_position = (text_position[0], text_position[1] + text_size[1] + 10)


    if step is None or location_stats is None:
        return frame
    
    info_text_lines = [
        f"Step: {step}",
        f"Pos: {[abs(int(location_stats['xpos'])) , abs(int(location_stats['ypos'])), abs(int(location_stats['zpos']))]}"
    ]
    
    for line in info_text_lines:
        put_colored_text(frame, line, text_position, FONT, FONT_SCALE, THICKNESS, LIGHT_ORANGE_COLOR)
        text_size = cv2.getTextSize(line, FONT, FONT_SCALE, THICKNESS)[0]
        text_position = (text_position[0], text_position[1] + text_size[1] + 10)

    return frame

def created_fitted_text_image(desired_width, text, thickness=2,
                              background_color=(255, 255, 255), text_color=(0, 0, 0), height_padding=20):
    """Create an image with text fitted to the desired width."""
    font_scale = 0.1
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    text_width, _ = text_size
    pad = desired_width // 5
    while text_width < desired_width - pad:
        font_scale += 0.1
        text_size, _ = cv2.getTextSize(text, FONT, font_scale, thickness)
        text_width, _ = text_size
    image = np.zeros((text_size[1] + 2 * height_padding, desired_width, 3), dtype=np.uint8)
    image[:] = background_color
    org = ((image.shape[1] - text_width) // 2, image.shape[0] - height_padding)
    return cv2.putText(image, text, org, FONT, font_scale, text_color, thickness)

