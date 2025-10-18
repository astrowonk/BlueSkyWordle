

### For full details, see my [blog post](/posts/wordle_bluesky/) and the [Github repository](https://github.com/astrowonk/BlueSkyWordle).

------

#### Solving Wordles with Social Media Shares

In short, I use the BlueSky API to find social media shares for a given Wordle number. i.e.

```
Wordle 1,575 5/6

🟩🟨⬜⬜🟨
🟩🟩🟩⬜⬜
⬜⬜⬜⬜⬜
🟨⬜⬜⬜🟩
🟩🟩🟩🟩🟩
```

Because different possible solutions have different possible score lines (and different likelihood of these lines), I can "solve" the Wordle and figure out the solution just by analyzing a few hundred social media shares. Finding the right heuristics to solve every Wordle involved a lot of post-hoc metrics, but now it works.

This web app explores the several hundred Wordles I have solved using shares from Blue Sky. Metrics used to rank the candidates:

#### **Fraction found.** 

What fraction of the candidate word's total possible patterns were found in the social media shares? (Note: this was upwards of 90% in Twitter data when I had 5000 or more posts, now it's usually in the 50-60% range)

#### **KS Statistic based on Opener Frequency**

This is the [two-sample Kolmogorov-Smirnov test statistic from SciPy](https://docs.scipy.org/doc/scipy-1.11.4/reference/generated/scipy.stats.ks_2samp.html). "The null hypothesis is that the two distributions are identical" - i.e. I'm comparing the prevalence of the score patterns in the *first guess* against the NYT's data of what people actually guess first. 
  
As you may [have heard](https://www.nytimes.com/2022/09/01/crosswords/wordle-starting-words-adieu.html?unlocked_article_code=1.uE8.vhfy.1tUiLm_4i5HY&smid=url-share), certain openers are consistently popular (*adieu*, *radio*, *crane*, etc.). This data is shown on the NYT [Wordlebot](https://www.nytimes.com/interactive/2022/upshot/wordle-bot.html), and can be extracted to get the counts of each first guess. I generate the frequency of the *score patterns* for each candidate these guesses correspond to, then compare to what first guess patterns I observe.

This tells me "*is this solution consistent with the patterns of the first guesses, knowing that certain first guesses are consistently popular*.""

One flaw is that I just extracted this for some relatively recent random day - it may not work perfectly in the past nor is it guaranteed to work in the future if opener patterns change.

#### **Impossible Pattern Count**

Certain patterns just can't occur for particular solutions. For example if the solution was `purse` the pattern can never be `🟨⬛🟩🟨🟩`. I use the [allowable word list](https://gist.github.com/dracos/dd0668f281e685bad51479e5acaadb93) to figure out if a pattern is possible or not, and I keep count of how many impossible patterns there are for each candidate as I iterate through all possible guesses and found patterns.

This should be *zero* for the actual solution, but in practice sometimes some errant score share shows an impossible pattern. Because I am not using any minimum count, I had to be more rigorous at filtering out random non-NYT wordle shares, like from `wordle.at` or other sites, etc - and those are other changes I made to make this code work. But sometimes there's a totally normal looking share that has an impossible pattern for the actual solution. Presumably someone playing an old version of the game?

Also, sometimes one ore more *incorrect* solutions has an impossible pattern count of 0.