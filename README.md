# TetrisAI-MB

At every step, the algorithm checks for every (rotation, column) pair and scores them using the 4 features and chooses the best pair. Using this pair, it drops the current piece to the chosen column with the chosen rotation.

Scoring is done taking the weighted average of the 4 features. The weights are constant. Different weights result in different play styles. For example, if the weight of holes features is increased, the resulting playstyle favors making long "towers" to avoid creating any hole.


### Running the code

No externals needed. Just run the gui.py file and watch the algorithm play.