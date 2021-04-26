# Dynamic-NMF topic-model on AWS SageMaker.

This repository includes code for implementating a dynamic-NMF topic-model on Amazon's SageMaker. The application of this implementation is to model topic distributions from the collection of all recorded speeches in the US House of Representatives and Senate (currently 1982-2020). The dynamic-NMF topic model approach was first introduced and developed by Derek Greene and James Cross (2017) for the analysis of two years of speeches (~270k speeches) from the European Parliament.

To scale up the dynamic-NMF for the size of the US congressional record, we turn to Amazon SageMaker.



- Greene, D., & Cross, J. (2017). Exploring the Political Agenda of the European Parliament Using a Dynamic Topic Modeling Approach. Political Analysis, 25(1), 77-94. https://www.jstor.org/stable/26563293


## Repository Content:
- containers: Docker containers for bring your own algorithm in AWS
- notebooks: jupyter notebooks for testing
- data: sample data for testing
