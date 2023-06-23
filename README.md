# Quantile-Hedging
Project made for Advanced Financial Engineering course during 2022/23 summer semester.
## What is Quantile Hedging?
The concept of Quantile Hedging comes from the idea of $\Delta$-hedging options in Black Scholes market assuptions without using the entire capital required 
for a full hedge(Black Scholes price of an option). Then, from a theoretical perspective we cannot secure ourselves from all scenarios and therefore
we only consider a given quantile of possibilities to be prepared for. In this circumstances, for given ammount of money (<Black-Scholes requirements for a full hedge) 
you can optimize the quantile itself (be prepared to as many plausible scenarios as possible) or additionally consider how much capital was missing in those not considered scenarios.
Idea behind Quantile Hedging approach is heavily based on a theoretical paper: *Quantile Hedging* by Hans Follmer and Peter Leukert.
## Short Decription of the project
Task was to implement Quantile Hedging pricing model together with $\Delta$-hedging simulations for different level of capital. 
We considered 2 optimization metrics: successfull hedge probability and success ratio. Vanilla options were mainly in scope of our calculations
but different non-path dependent payoffs can also be implemented through classes we made. Additionally, options for non-tradable underlyings
were also considered- here the approach was more complicated than in the original paper of Hans Follmer and Peter Leukert.
## Technical description
Project was made with object-programming approach. We build 3 main classes interacting with each other: Underlying, Option and Trader. Quantitative
functions implementing Quantile Hedging methods were constructed without use of objects. Jupyter notebook was created as the final result of the project- 
it consists of a broad analysis on the entire topic, together with plenty analyses on the performence, sensivity and correctness of our methods. The report 
is unfortunately in polish, but all of the code + theoretical formulas behind it can easily be followed also by non-polish speakers.
