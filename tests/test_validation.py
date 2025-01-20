from torch.utils.data import DataLoader
from validation.validator import TextImageValidator
from validation.dataset import ImageTextDataset


def test_validators():
    dataset = ImageTextDataset(csv_file='data/challenge_set.csv', image_dir='data/images')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))

    clip_validator = TextImageValidator(method='clip', model_name="openai:ViT-L-14-336", debug=True)
    clip_output = clip_validator.evaluate(batch['image_path'], batch['text'])
    print(clip_output)

    blip_validator = TextImageValidator(method='blip', model_name="blip2-itm", debug=True)
    blip_output = blip_validator.evaluate(batch['image_path'], batch['text'])
    print(blip_output)

    vqa_validator = TextImageValidator(method='vqa', model_name="clip-flant5-xl", debug=True)
    vqa_output = vqa_validator.evaluate(batch['image_path'], batch['text'])
    print(vqa_output)
