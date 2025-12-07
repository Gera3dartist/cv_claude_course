"""Character recognition using template matching with structural features."""
import cv2
import numpy as np


def create_char_templates() -> tuple[dict, dict]:
    """Create templates and precompute their features.

    Returns:
        Tuple of (templates dict, features dict)
    """
    templates = {}
    features = {}
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    for char in chars:
        template = np.zeros((60, 45), dtype=np.uint8)
        cv2.putText(template, char, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
        templates[char] = template
        features[char] = extract_structural_features(template)

    return templates, features


def extract_structural_features(char_img: np.ndarray) -> dict:
    """Extract structural features from character image."""
    h, w = char_img.shape

    # Normalize
    if char_img.max() > 0:
        normalized = (char_img > char_img.max() * 0.5).astype(np.uint8) * 255
    else:
        normalized = char_img

    total = max(normalized.sum(), 1)

    # Horizontal profile (top, mid, bottom)
    top_third = normalized[:h//3, :].sum() / total
    mid_third = normalized[h//3:2*h//3, :].sum() / total
    bot_third = normalized[2*h//3:, :].sum() / total

    # Top bar detection
    top_region = normalized[:h//5, :]
    top_density = top_region.sum() / max(normalized[:h//2, :].sum(), 1)
    has_top_bar = top_density > 0.25

    # Width at different heights
    top_width = np.count_nonzero(normalized[h//6, :])
    mid_width = np.count_nonzero(normalized[h//2, :])

    return {
        'h_profile': (top_third, mid_third, bot_third),
        'has_top_bar': has_top_bar,
        'top_width': top_width,
        'mid_width': mid_width,
    }


def distinguish_T_from_1(char_img: np.ndarray) -> str:
    """Distinguish between T and 1 based on structural features.

    Args:
        char_img: Grayscale character image (white char on black background)

    Returns:
        'T' or '1' based on structural analysis
    """
    # TODO(human): Implement the logic to distinguish T from 1
    #
    # Hints:
    # - T has a horizontal bar at the top → high density in top rows
    # - T is wider at the top than in the middle
    # - 1 has relatively uniform width throughout
    # - 1's mass is concentrated in the vertical center
    #
    # Useful operations:
    # - char_img[:h//5, :].sum()  → sum of top 20% of image
    # - np.count_nonzero(char_img[row, :])  → width at specific row
    # - char_img.shape → (height, width)
    #
    # Return 'T' if it looks like T, otherwise '1'

    return '?'  # Replace with your implementation


def match_character(char_img: np.ndarray, templates: dict, features: dict) -> str:
    """Match character using template matching + structural features."""
    char_resized = cv2.resize(char_img, (45, 60))
    char_features = extract_structural_features(char_resized)

    scores = {}

    for char, template in templates.items():
        # Template matching
        result = cv2.matchTemplate(char_resized, template, cv2.TM_CCOEFF_NORMED)
        template_score = result.max()

        # Feature similarity
        tmpl_feat = features[char]
        h_diff = sum(abs(a - b) for a, b in zip(
            char_features['h_profile'], tmpl_feat['h_profile']
        ))

        # Top bar penalty
        top_bar_penalty = 0.2 if char_features['has_top_bar'] != tmpl_feat['has_top_bar'] else 0

        # Combined score
        feature_score = 1 - h_diff - top_bar_penalty
        scores[char] = 0.6 * template_score + 0.4 * feature_score

    # Get top 2 candidates
    sorted_chars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_char, best_score = sorted_chars[0]
    second_char, second_score = sorted_chars[1]

    # If top candidates are T and 1 and scores are close, use special resolver
    if {best_char, second_char} == {'T', '1'} and (best_score - second_score) < 0.15:
        return distinguish_T_from_1(char_resized)

    return best_char if best_score > 0.4 else '?'


def recognize_plate(char_images: list) -> str:
    """Recognize all characters in a license plate.

    Args:
        char_images: List of segmented character images

    Returns:
        Recognized plate string
    """
    templates, features = create_char_templates()

    result = []
    for char_img in char_images:
        recognized = match_character(char_img, templates, features)
        result.append(recognized)

    return ''.join(result)


# Debug visualization
def debug_confusion(char_img: np.ndarray, templates: dict, features: dict):
    """Visualize why a character might be confused."""
    import matplotlib.pyplot as plt

    char_resized = cv2.resize(char_img, (45, 60))
    char_feat = extract_structural_features(char_resized)

    print("Character features:")
    print(f"  H-profile: {[f'{x:.2f}' for x in char_feat['h_profile']]}")
    print(f"  Has top bar: {char_feat['has_top_bar']}")
    print(f"  Top width: {char_feat['top_width']}, Mid width: {char_feat['mid_width']}")

    # Compare with T and 1
    for char in ['T', '1']:
        print(f"\n{char} template features:")
        print(f"  H-profile: {[f'{x:.2f}' for x in features[char]['h_profile']]}")
        print(f"  Has top bar: {features[char]['has_top_bar']}")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(char_resized, cmap='gray')
    axes[0].set_title('Input Character')
    axes[1].imshow(templates['T'], cmap='gray')
    axes[1].set_title('T Template')
    axes[2].imshow(templates['1'], cmap='gray')
    axes[2].set_title('1 Template')
    plt.show()
