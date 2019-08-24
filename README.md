# final_year

Over the last few years, for the auto insurance claims process, improvements in the First Notice of Loss and rapidity in the investigation and evaluation of claims could drive significant values by reducing loss adjustment expense. Image based vehicle insurance processing is an important area with large scope for automation. In this report we are going to consider the problem of car damage classification, where some of the categories can be fine-granular. We explore deep learning based techniques for this purpose. Success in this will allow some cases to proceed without human surveyor, while others to proceed more efficiently, thus ultimately shortening the time between the first Notice of Loss and the final payout.
In the proposed car damage classification model initially,
we try directly training a CNN. However, due to small set of labeled data, it does not work well.
Then, we explore the effect of domain specific pre-training followed by fine-tuning. Finally,
we experiment with transfer learning and ensemble learning. Experimental results show that transfer learning works better than domain 
specific fine-tuning. We achieve accuracy of 89.5% with combination of transfer and ensemble learning.Followed by 
the estimate damage cost calculation according the prediction of damaged car parts.
