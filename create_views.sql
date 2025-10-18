CREATE INDEX IF NOT EXISTS idxsolution on hashed_solutions(wordle_id);

CREATE INDEX IF NOT EXISTS pattern_idx1 on patterns(target, pattern);

CREATE INDEX IF NOT EXISTS pattern_idx2 on patterns(target, guess);

CREATE INDEX IF NOT EXISTS idx_wordleid on solutions(wordle_id);

CREATE INDEX IF NOT EXISTS idx_patterncounts on pattern_counts(wordle_id);

CREATE VIEW IF NOT EXISTS ranked_view as
select
	*,
	rank() over (
		partition by wordle_id
		order by
			metric_sum asc
	) metric_sum_rank
from
	(
		select
			wordle_id,
			word,
			fraction_found_rank + norm_score_rank + kstat_rank + power(1 + impossible_pattern_count, 2) as metric_sum,
			is_solution,
			fraction_found_rank,
			norm_score_rank,
			kstat_rank,
			impossible_pattern_count
		from
			(
				select
					*,
					rank() over (
						partition by wordle_id
						order by
							fraction_found desc
					) fraction_found_rank,
					rank() over (
						partition by wordle_id
						order by
							norm_score desc
					) norm_score_rank,
					rank() over (
						partition by wordle_id
						order by
							kstatistic asc
					) kstat_rank
				from
					solutions
			)
	)
	/* ranked_view(wordle_id,word,metric_sum,is_solution,fraction_found_rank,norm_score_rank,kstat_rank,impossible_pattern_count,metric_sum_rank) */
;