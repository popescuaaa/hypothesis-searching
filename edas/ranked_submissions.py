# Combine all submissions based on their accuracies into one submission

import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    submissions_folder = "submissions/"
    submissions = os.listdir(submissions_folder)

    print("Submissions: ", submissions)
    ranks_map = {
        'alexnet_20_0.001_use_scheduler_True_prob_samp_True.csv': 1.04 / 100,
         'vgg_20_0.001_use_scheduler_True_prob_samp_True.csv': 1.02 / 100,
         'densenet_20_0.001_use_scheduler_True_prob_samp_True.csv': 1.36 / 100,
         'inception_20_0.001_use_scheduler_True_prob_samp_True.csv': 1.06 / 100,
         'resnet_20_0.001_use_scheduler_True_prob_samp_True.csv': 0.88 / 100,
         'squeezenet_20_0.001_use_scheduler_True_prob_samp_True.csv': 1.08 / 100,
    }

    # Convert the ranks to probabilities
    total = sum(ranks_map.values())
    for key in ranks_map:
        ranks_map[key] /= total

    print("Ranks: ", ranks_map)

    weighted_submissions = []
    for submission in submissions:
        submission_df = pd.read_csv(submissions_folder + submission)
        submission_df["label"] = submission_df["label"] * ranks_map[submission]
        weighted_submissions.append(submission_df)

    # Combine the submissions
    combined_submission = weighted_submissions[0]
    for i in range(1, len(weighted_submissions)):
        combined_submission["label"] += weighted_submissions[i]["label"]
        # Make sure its integer
        combined_submission["label"] = combined_submission["label"].astype(int)

    # Save the combined submission
    combined_submission.to_csv(submissions_folder + "combined_submission.csv", index=False)