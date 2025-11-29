# dataset_info.py
from typing import Dict, Any, Optional

DATASET_INFO_MAP = {
    'har70plus': {
        'channels': 6, 
        'time_steps': 500, 
        'num_classes': 7,
        'description': 'Chest (sternum) sensor data, including fine-grained daily activities such as brushing teeth and chopping vegetables'
    },
    'MotionSense': {
        'channels': 6, 
        'time_steps': 500, 
        'num_classes': 6,
        'description': 'Front right trouser pocket sensor data, including basic activities such as walking, jogging and climbing stairs'
    },
    'w-HAR': {
        'channels': 6, 
        'time_steps': 2500, 
        'num_classes': 7,
        'description': 'Left wrist sensor data, including walking, running, jumping and other office and daily movements'
    },
    'WISDM': {
        'channels': 6,
        'time_steps': 200,
        'num_classes': 18,
        'description': 'A set of data collected based on sensors placed in pants pockets and wrists, including fine-grained actions such as walking, running, going up and down stairs, sitting and standing.'
    },
    'Harth': {
        'channels': 6,
        'time_steps': 500,
        'num_classes': 12,
        'description': 'A set of sensor data based on the right thigh and lower back, including cooking/cleaning, Yoga/weight lifting, walking on the flat/stairs, etc.'
    },
    'USCHAD': {
        'channels': 6,
        'time_steps': 1000,
        'num_classes': 12,
        'description': 'A group of sensing data based on the right front hip, including walking, running, going upstairs, going downstairs, jumping, sitting, standing, sleeping and taking the elevator.'
    },
    'UTD-MHAD': {
        'channels': 6,
        'time_steps': 300,
        'num_classes': 27,
        'description': 'A group of sensing data based on the right wrist or right thigh, including waving, punching, clapping, jumping, push ups and other actions.'
    },
    'DSADS': {
        'channels': 45,
        'time_steps': 125,
        'num_classes': 19,
        'description': 'A group of sensing data based on trunk, right arm, left arm, right leg and left leg, including whole body and local actions such as sitting and relaxing, using computer'
    },
    'realworld': {
        'channels': 42,
        'time_steps': 500,
        'num_classes': 8,
        'description': 'The HAR dataset collected from chest, forearm, head, and lower limb sensors includes activities like standing, sitting, and lying down and so on.'
    },
    'Shoaib': {
        'channels': 30,
        'time_steps': 500,
        'num_classes': 7,
        'description': 'The HAR dataset collected from right jeans pocket, waist, and upper arm sensors (and so on) includes walking, running, and cycling activities (and so on).'
    },
    'TNDA-HAR': {
        'channels': 30,
        'time_steps': 500,
        'num_classes': 8,
        'description': 'The HAR dataset collected from upper arm, lower leg, and waist (and so on) positions includes walking, jogging, jumping, and arm-raising motions (and so on).'
    },
    'UCIHAR': {
        'channels': 9,
        'time_steps': 128,
        'num_classes': 6,
        'description': 'The waist-mounted HAR dataset includes walking, upstairs, and downstairs activities (and so on).'
    },
    'ut-complex': {
        'channels': 12,
        'time_steps': 500,
        'num_classes': 13,
        'description': 'The waist-and-pocket HAR dataset includes mopping, window cleaning, and carrying activities (and so on).'
    },
    'Wharf': {
        'channels': 3,
        'time_steps': 320,
        'num_classes': 14,
        'description': 'The wrist-based HAR dataset includes loading/unloading, forklift operation, and walking activities (and so on).'
    },
    'Mhealth': {
        'channels': 23,
        'time_steps': 500,
        'num_classes': 12,
        'description': 'The chest-and-wrist HAR dataset includes standing, sitting, supine, and lateral lying postures (and so on).'
    },
    'MMAct': {
        'channels': 9,
        'time_steps': 250,
        'num_classes': 35,
        'description': 'The pants-pocket HAR dataset includes walking, running, and cycling activities (and so on).'
    },
    'Opp_g': {
        'channels': 45,
        'time_steps': 300,
        'num_classes': 4,
        'description': 'The right-arm/left-arm/right-shoe HAR dataset includes door opening/closing and cup grasping activities (and so on).'
    },
    'PAMAP': {
        'channels': 27,
        'time_steps': 1000,
        'num_classes': 12,
        'description': 'The wrist-and-chest (and so on) HAR dataset includes walking, running, and cycling activities (and so on).'
    }
}

def get_dataset_info(dataset_name: str) -> dict:
    """Retrieve information for the specified dataset"""
    return DATASET_INFO_MAP.get(dataset_name, {})
