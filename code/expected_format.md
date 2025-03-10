# Expected Data Formats
## Dagstuhl-15512-ArgQuality
**Expert annotations (Wachsmuth et al., 2017: Computational argumentation quality assessment in natural language)**

A CSV with the following header:
```
annotator	argumentative	overall quality	local acceptability	appropriateness	arrangement	clarity	cogency	effectiveness	global acceptability	global relevance	global sufficiency	reasonableness	local relevance	credibility	emotional appeal	sufficiency	argument	#id	issue	stance
```
Note that the separator is a tab.

**Crowd annotations (Wachsmuth et al., 2017: Argument quality assessment: Theory vs. practice)**
A CSV with the following header:
```
_unit_id,_created_at,_id,_started_at,_tainted,_channel,_trust,_worker_id,_country,_region,_city,_ip,_please_provide_us_with_anything_else_that_affected_your_assessment_of_the_arguments_and_that_is_not_covered_by_the_above_dimensions,how_do_you_rate_the_overall_quality_of_the_authors_argumentation,how_would_you_rate_the_acceptability_of_the_premises_of_the_authors_arguments,how_would_you_rate_the_appropriateness_of_the_style_of_the_authors_argumentation,how_would_you_rate_the_arrangement_of_the_authors_argumentation,how_would_you_rate_the_clarity_of_the_style_of_the_authors_argumentation,how_would_you_rate_the_cogency_of_the_authors_arguments_,how_would_you_rate_the_effectiveness_of_the_authors_argumentation,how_would_you_rate_the_global_acceptability_of_the_authors_argumentation,how_would_you_rate_the_global_relevance_of_the_authors_argumentation,how_would_you_rate_the_global_sufficiency_of_the_authors_argumentation,how_would_you_rate_the_reasonableness_of_the_authors_argumentation,how_would_you_rate_the_relevance_of_the_premises_to_the_authors_conclusions_,how_would_you_rate_the_success_of_the_authors_argumentation_in_creating_credibility,how_would_you_rate_the_success_of_the_authors_argumentation_in_making_an_emotional_appeal_,how_would_you_rate_the_sufficiency_of_the_premises_of_the_authors_arguments_,content,exp_id,how_do_you_rate_the_overall_quality_of_the_authors_argumentation_gold,how_would_you_rate_the_acceptability_of_the_premises_of_the_authors_arguments_gold,how_would_you_rate_the_appropriateness_of_the_style_of_the_authors_argumentation_gold,how_would_you_rate_the_arrangement_of_the_authors_argumentation_gold,how_would_you_rate_the_clarity_of_the_style_of_the_authors_argumentation_gold,how_would_you_rate_the_cogency_of_the_authors_arguments__gold,how_would_you_rate_the_effectiveness_of_the_authors_argumentation_gold,how_would_you_rate_the_global_acceptability_of_the_authors_argumentation_gold,how_would_you_rate_the_global_relevance_of_the_authors_argumentation_gold,how_would_you_rate_the_global_sufficiency_of_the_authors_argumentation_gold,how_would_you_rate_the_reasonableness_of_the_authors_argumentation_gold,how_would_you_rate_the_relevance_of_the_premises_to_the_authors_conclusions__gold,how_would_you_rate_the_success_of_the_authors_argumentation_in_creating_credibility_gold,how_would_you_rate_the_success_of_the_authors_argumentation_in_making_an_emotional_appeal__gold,how_would_you_rate_the_sufficiency_of_the_premises_of_the_authors_arguments__gold,issue,stance
```

**Novice annotations (Mirzakhmedova et al., 2024: Are large language models reliable argument quality annotators?)**

JSON files containing the raw annotations per annotator with the naming scheme:
```
'argquality23-groupXX-memberYY-<date + id>.json'
```

## GAQ Corpus
**Expert and crowd annotations (Lauscher et al., 2020: Rhetoric, logic, and dialect: Advancing theory-based argument quality assessment in natural language processing)**

CSVs per annotator group and domain. For the analysis in this work, only
``` debate_forums_crowd.csv``` and ```debate_forums_experts.csv``` are relevant.

The headers should be:
```
id,cogency,cogency_mean,effectiveness,effectiveness_mean,reasonableness,reasonableness_mean,overall,overall_mean,argumentative,argumentative_majority,text,title
```