# SegmentingMars

My motivation and inspiration for pursuing this project started when I was reading Meta AI’s paper on SAM (Segment Anything Model).

While reading the research paper, I wondered if this technique could be used to identify areas of interest/avoidance in autonomously driving the Mars Rovers. After conversations, I learned that semantic labelling is a necessity, so I looked at Grounding DINO. SAM and Grounding DINO are compatible in that they both take a pixel by pixel approach. SAM is has an individualized pixel approach while Grounding DINO is more cluster pixel approach. By having Grounding DINO run off of SAM's outputs, it decreased the overall operational time when comparing it to an only Grounding DINO approach.

This project had two versions. 

1. SAM only: Because SAM was trained on Earth data, the model struggled with Mars images
2. SAM + Grounding DINO: After conversations with a robotics lead at JPL, I learned that semantic labeling was a necessity, so I researched several semantic labelling models and decided on Grounding DINO. 

Moving Forward:
1. Fine Tuning: The model’s performance was subpar. I don't have any subject matter expertise to develop fine tuning mechanisms
2. Grounding DINO Alternatives: Exploring other semantic labeling models with a diverse vocabulary

# Requirements for model
1. Need to download SAM (https://github.com/facebookresearch/segment-anything ) and Grounding DINO (https://github.com/IDEA-Research/GroundingDINO).
2. Download checkpoints for respective models
3. Gather images that you'd like to try and feed into model

After Downloading the necessary models, run 'SegmentingMars.py'
