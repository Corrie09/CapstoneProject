## okCupid data to do: 

age : ok 
status : 5 distinct categories (10unknown)
sex : ok m and f
orientation : ok just 3 categories
bodytype : many NaN (or rather not say) and few that are the same like thin and skinny
diet : a lot of NaN a lot of the same like mostly anything  and anything or vegetariand and strictly vegetarian
drinks : ok but kind of equal ex: often and very often
drugs : ok just 3 but a lot missing 
education : bad, a lot of almost the same and also some put working here (so maybe convert to the working column)
ethnicity : very bad, a lot of different things that are the same like: black , white and white, black ... 
height : ok just numbers 
income : ok is in categories but a lot missing (indicated as -1)
job: ok but a lot of NaN and a bit weird categorized
last online : unusefull 
location : city + state but i would say unusefull 
offspring : bit all over the place, (has kids and has a kid) a lot of combinations but can be reduced pretty easily
pets : bad because is having as well as liking pets 
religion : annoying because is religion + how serious you take it
sign : very bad done 
smokes : good 
speaks : language is bad because a lot of different combinations (languague + how good at it)
essays : just thousands of different kind sentences about themselves


essay0- My self summary
essay1- What I’m doing with my life
essay2- I’m really good at
essay3- The first thing people usually notice about me
essay4- Favorite books, movies, show, music, and food
essay5- The six things I could never do without
essay6- I spend a lot of time thinking about
essay7- On a typical Friday night I am
essay8- The most private thing I am willing to admit
essay9- You should message me if...



total dataset: 
- 30 columns (10 about essays)
- only 3 columns with numeric data 

## Option B — Information Disclosure under Uncertainty

**Idea**  
Users face uncertainty about others’ preferences and decide *how much to reveal* to reduce search frictions.

**Economic concept**  
*Value of information / Bayesian learning*:contentReference[oaicite:0]{index=0}

**Data angle**  
Probability of filling out optional fields (`pets`, `diet`, `religion`, etc.) reflects willingness to share private information.

**Predictions**  
1. Users who disclose more optional attributes have more complete essays.  
2. Disclosure correlates with demographic factors (education, age, etc.) and could be higher for users with less common traits (scarcity effect).  

**Empirical model**  
DisclosureIndex = (# of filled optional fields / total optional fields)  
DisclosureIndex ~ age + education + sex + orientation + scarcity_index  

---

## Option C — Market Composition and Scarcity

**Idea**  
In a two-sided market, some traits are scarce (e.g., few women in a given orientation group). Scarcity shapes signaling effort.

**Economic concept**  
*Matching equilibrium* and *market thickness*:contentReference[oaicite:1]{index=1}

**Data angle**  
Compare relative proportions of genders/orientations within locations → see if scarce groups invest less in signaling effort (because they already get attention).

**Predictions**  
1. Users in scarce groups (e.g., straight women, gay men, etc.) write *shorter essays* or disclose less information.  
2. The relationship between scarcity and effort is negative.  

**Empirical model**  
word_count ~ scarcity_index + controls  

**Notes**  
`scarcity_index` = inverse of share of users of same sex/orientation in local sample.  

---

## Option D — Preferences and Homophily

**Idea**  
Examine how users describe themselves vs. what they say they’re looking for (`essay0`–`essay9`) to see patterns of similarity preferences.

**Economic concept**  
*Assortative matching* and *revealed preferences*:contentReference[oaicite:2]{index=2}

**Data angle**  
Compare text content or topics between self-descriptions and partner descriptions.

**Challenge**  
Requires text analysis (sentiment, topic modeling, or embeddings).

**Predictions**  
1. Education or occupation language clusters correlate with real education levels.  
2. People emphasize traits that are valued by the opposite sex group.

**Empirical model (conceptual)**  
similarity_score(self_description, partner_description) ~ demographics + preferences  




# real possabilities others

can build:

word_count = number of words across essays (signaling effort)

completion_rate = % of non-missing optional fields (diet, pets, offspring, etc.)

scarcity_index = inverse proportion of same-sex/orientation group

These can all become dependent or explanatory variables depending on your question.


project in general: 

1. You pick an economic phenomenon (a decision, a trade-off, a market).

2. You explain the incentives, information, and interactions behind it (the model).

3. You use data to illustrate or test those mechanisms.

my choice:
“How do people signal quality in dating markets when information is asymmetric?”

Theory:
People choose how much information to disclose (essays, education, preferences) based on signaling theory.
Revealing more information is costly but can increase perceived quality.

Empirical test:
Using OkCupid data, check whether users with higher education or income write longer essays or fill in more fields.


