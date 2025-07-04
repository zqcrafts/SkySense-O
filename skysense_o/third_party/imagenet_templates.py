# source: https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb

IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
    # 'A photo of a {} in the scene.',
]

# v1: 59.0875
IMAGENET_TEMPLATES_SELECT = [
    'itap of a {}.',
    'a bad photo of the {}.',
    'a origami {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
    'A photo of a {} in the scene',
]

# v2: 58.2584
# IMAGENET_TEMPLATES_SELECT = [
#     'itap of a {}',
#     'a bad photo of the {}',
#     'a origami {}',
#     'a photo of the large {}',
#     'art of the {}',
#     'a photo of the small {}',
#     'A photo of a {} in the scene',
# ]

# v3: 59.1006
# IMAGENET_TEMPLATES_SELECT = [
#     'itap of a {}.',
#     'a bad photo of the {}.',
#     'a origami {}.',
#     'a photo of the large {}.',
#     'art of the {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'A photo of a {} in the scene',
#     'itap of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a origami {} in the scene',
#     'a photo of the large {} in the scene',
#     'art of the {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
# ]

# v4: 59.8659
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'art of the {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'art of the {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
# ]

# v5: 59.9346
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'art of the {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'art of the {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
# ]

# v6: 60.6611
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'art of the {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'art of the {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
#
#     'There is a masked {} in the scene',
#     'There is the masked {} in the scene',
#     'This is a masked {} in the scene',
#     'This is the masked {} in the scene',
#     'This is one masked {} in the scene',
# ]

# v7: 60.4529
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'art of the {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'art of the {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
#
#     'There is a cropped {} in the scene',
#     'There is the cropped {} in the scene',
#     'This is a cropped {} in the scene',
#     'This is the cropped {} in the scene',
#     'This is one cropped {} in the scene',
#
#     'a cropped photo of the {}',
#     'a cropped photo of a {}',
#     'a cropped photo of one {}',
#
#     'There is a masked {} in the scene',
#     'There is the masked {} in the scene',
#     'This is a masked {} in the scene',
#     'This is the masked {} in the scene',
#     'This is one masked {} in the scene',
# ]

# v8: 60.7057
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#
#     'This is a masked photo of a {}',
#     'This is a masked photo of a small {}',
#     'This is a masked photo of a medium {}',
#     'This is a masked photo of a large {}',
#
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
#
#     'There is a masked {} in the scene',
#     'There is the masked {} in the scene',
#     'This is a masked {} in the scene',
#     'This is the masked {} in the scene',
#     'This is one masked {} in the scene',
# ]

# v9: 60.8775
# IMAGENET_TEMPLATES_SELECT = [
#     'a bad photo of the {}.',
#     'a photo of the large {}.',
#     'a photo of the small {}.',
#     'a cropped photo of a {}.',
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#
#     'This is a masked photo of a {}',
#     'This is a masked photo of a small {}',
#     'This is a masked photo of a medium {}',
#     'This is a masked photo of a large {}',
#
#     'This is a cropped photo of a {}',
#     'This is a cropped photo of a small {}',
#     'This is a cropped photo of a medium {}',
#     'This is a cropped photo of a large {}',
#
#     'A photo of a {} in the scene',
#     'a bad photo of the {} in the scene',
#     'a photo of the large {} in the scene',
#     'a photo of the small {} in the scene',
#     'a cropped photo of a {} in the scene',
#     'a photo of a masked {} in the scene',
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
#
#     'There is a masked {} in the scene',
#     'There is the masked {} in the scene',
#     'This is a masked {} in the scene',
#     'This is the masked {} in the scene',
#     'This is one masked {} in the scene',
# ]

# v9
IMAGENET_TEMPLATES_SELECT_CLIP = [
    'a bad photo of the {}.',
    'a photo of the large {}.',
    'a photo of the small {}.',
    'a cropped photo of a {}.',
    'This is a photo of a {}',
    'This is a photo of a small {}',
    'This is a photo of a medium {}',
    'This is a photo of a large {}',
    'This is a masked photo of a {}',
    'This is a masked photo of a small {}',
    'This is a masked photo of a medium {}',
    'This is a masked photo of a large {}',
    'This is a cropped photo of a {}',
    'This is a cropped photo of a small {}',
    'This is a cropped photo of a medium {}',
    'This is a cropped photo of a large {}',
    'A photo of a {} in the scene',
    'a bad photo of the {} in the scene',
    'a photo of the large {} in the scene',
    'a photo of the small {} in the scene',
    'a cropped photo of a {} in the scene',
    'a photo of a masked {} in the scene',
    'There is a {} in the scene',
    'There is the {} in the scene',
    'This is a {} in the scene',
    'This is the {} in the scene',
    'This is one {} in the scene',
    'There is a masked {} in the scene',
    'There is the masked {} in the scene',
    'This is a masked {} in the scene',
    'This is the masked {} in the scene',
    'This is one masked {} in the scene',
]

# v10, for comparison
# IMAGENET_TEMPLATES_SELECT_CLIP = [
#     'a photo of a {}.',
#
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#
#     'This is a photo of a {}',
#     'This is a photo of a small {}',
#     'This is a photo of a medium {}',
#     'This is a photo of a large {}',
#
#     'a photo of a {} in the scene',
#     'a photo of a {} in the scene',
#
#     'There is a {} in the scene',
#     'There is the {} in the scene',
#     'This is a {} in the scene',
#     'This is the {} in the scene',
#     'This is one {} in the scene',
# ]

ViLD_templates = [
    'There is {article} {category} in the scene.',
    'There is the {category} in the scene.',
    'a photo of {article} {category} in the scene.',
    'a photo of the {category} in the scene.',
    'a photo of one {category} in the scene.', 'itap of {article} {category}.',
    'itap of my {category}.', 'itap of the {category}.',
    'a photo of {article} {category}.', 'a photo of my {category}.',
    'a photo of the {category}.', 'a photo of one {category}.',
    'a photo of many {category}.', 'a good photo of {article} {category}.',
    'a good photo of the {category}.', 'a bad photo of {article} {category}.',
    'a bad photo of the {category}.', 'a photo of a nice {category}.',
    'a photo of the nice {category}.', 'a photo of a cool {category}.',
    'a photo of the cool {category}.', 'a photo of a weird {category}.',
    'a photo of the weird {category}.', 'a photo of a small {category}.',
    'a photo of the small {category}.', 'a photo of a large {category}.',
    'a photo of the large {category}.', 'a photo of a clean {category}.',
    'a photo of the clean {category}.', 'a photo of a dirty {category}.',
    'a photo of the dirty {category}.',
    'a bright photo of {article} {category}.',
    'a bright photo of the {category}.',
    'a dark photo of {article} {category}.', 'a dark photo of the {category}.',
    'a photo of a hard to see {category}.',
    'a photo of the hard to see {category}.',
    'a low resolution photo of {article} {category}.',
    'a low resolution photo of the {category}.',
    'a cropped photo of {article} {category}.',
    'a cropped photo of the {category}.',
    'a close-up photo of {article} {category}.',
    'a close-up photo of the {category}.',
    'a jpeg corrupted photo of {article} {category}.',
    'a jpeg corrupted photo of the {category}.',
    'a blurry photo of {article} {category}.',
    'a blurry photo of the {category}.',
    'a pixelated photo of {article} {category}.',
    'a pixelated photo of the {category}.',
    'a black and white photo of the {category}.',
    'a black and white photo of {article} {category}.',
    'a plastic {category}.', 'the plastic {category}.', 'a toy {category}.',
    'the toy {category}.', 'a plushie {category}.', 'the plushie {category}.',
    'a cartoon {category}.', 'the cartoon {category}.',
    'an embroidered {category}.', 'the embroidered {category}.',
    'a painting of the {category}.', 'a painting of a {category}.'
]