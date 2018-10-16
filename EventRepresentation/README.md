# Event representation learning

## Role-Factor Tensor Network

* Paper: [Weber et al. (2018). Event Representations with Tensor-based Compositions](https://arxiv.org/pdf/1711.07611.pdf)
* PyTorch version: 0.4.0
* To run, configure the paths in `run_rtf_net.sh` and run the script. NB: place the glove `.txt` file in the same folder as all the `.py` scripts. Go to this [site](https://nlp.stanford.edu/projects/glove/) to download glove embeddings (download `glove.6B.zip`). 

## Wikipedia Dump Preprocessing (TACC running instructions)

* Put the following files under the same folder (the last two files are in the [drive](https://drive.google.com/open?id=132FGfmOHtORnHjSVWL9WocoRFEwFocm0))
  * `wiki_event_extraction.py`
  * `wiki_batch_0x` (00 to 18)
  * `engmalt.linear-1.7.mco`
  * `ollie-app-latest.jar`
* Edit the batch runner `wiki_batch_0x` for the `.bz` wiki file to be parsed (to parse `wiki_08.bz`, for instance, replace all the 00's in the file with 08). Also edit the email to yours to receive status report.
* Run `sbatch wiki_batch_0x` (00 to 18).
* To check the status of your job with `showq -u`.

**NB**: replace the `UT-DEFT` in the line `SBATCH -A ...` to `cs395t-f18`. I'm running UT-DEFT because I'm funded through that project.

## Wikipedia Job Assignments

* Su: 00 to 07 (8 files in total)
* Elisa: 08 to 13 (6 files in total)
* Brahma: 14 to 18 (5 files in total)

**NB**: the jobs should be run in `gpu-long` for 72 hours. The max capacity on Maverick is 8 jobs for each user. 
