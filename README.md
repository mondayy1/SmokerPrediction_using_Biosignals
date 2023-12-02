# SmokerPrediction_using_Biosignals
Let's create a model that predicts smoking using bio-signals.

## Datasets
https://www.kaggle.com/datasets/gauravduttakiit/smoker-status-prediction-using-biosignals/data

## Dataset Overview
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>height(cm)</th>
      <th>weight(kg)</th>
      <th>waist(cm)</th>
      <th>eyesight(left)</th>
      <th>eyesight(right)</th>
      <th>hearing(left)</th>
      <th>hearing(right)</th>
      <th>systolic</th>
      <th>relaxation</th>
      <th>...</th>
      <th>HDL</th>
      <th>LDL</th>
      <th>hemoglobin</th>
      <th>Urine protein</th>
      <th>serum creatinine</th>
      <th>AST</th>
      <th>ALT</th>
      <th>Gtp</th>
      <th>dental caries</th>
      <th>smoking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35</td>
      <td>170</td>
      <td>85</td>
      <td>97.0</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>1</td>
      <td>1</td>
      <td>118</td>
      <td>78</td>
      <td>...</td>
      <td>70</td>
      <td>142</td>
      <td>19.8</td>
      <td>1</td>
      <td>1.0</td>
      <td>61</td>
      <td>115</td>
      <td>125</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>175</td>
      <td>110</td>
      <td>110.0</td>
      <td>0.7</td>
      <td>0.9</td>
      <td>1</td>
      <td>1</td>
      <td>119</td>
      <td>79</td>
      <td>...</td>
      <td>71</td>
      <td>114</td>
      <td>15.9</td>
      <td>1</td>
      <td>1.1</td>
      <td>19</td>
      <td>25</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>45</td>
      <td>155</td>
      <td>65</td>
      <td>86.0</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>1</td>
      <td>1</td>
      <td>110</td>
      <td>80</td>
      <td>...</td>
      <td>57</td>
      <td>112</td>
      <td>13.7</td>
      <td>3</td>
      <td>0.6</td>
      <td>1090</td>
      <td>1400</td>
      <td>276</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>45</td>
      <td>165</td>
      <td>80</td>
      <td>94.0</td>
      <td>0.8</td>
      <td>0.7</td>
      <td>1</td>
      <td>1</td>
      <td>158</td>
      <td>88</td>
      <td>...</td>
      <td>46</td>
      <td>91</td>
      <td>16.9</td>
      <td>1</td>
      <td>0.9</td>
      <td>32</td>
      <td>36</td>
      <td>36</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>165</td>
      <td>60</td>
      <td>81.0</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>1</td>
      <td>1</td>
      <td>109</td>
      <td>64</td>
      <td>...</td>
      <td>47</td>
      <td>92</td>
      <td>14.9</td>
      <td>1</td>
      <td>1.2</td>
      <td>26</td>
      <td>28</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38979</th>
      <td>40</td>
      <td>165</td>
      <td>60</td>
      <td>80.0</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>1</td>
      <td>1</td>
      <td>107</td>
      <td>60</td>
      <td>...</td>
      <td>61</td>
      <td>72</td>
      <td>12.3</td>
      <td>1</td>
      <td>0.5</td>
      <td>18</td>
      <td>18</td>
      <td>21</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38980</th>
      <td>45</td>
      <td>155</td>
      <td>55</td>
      <td>75.0</td>
      <td>1.5</td>
      <td>1.2</td>
      <td>1</td>
      <td>1</td>
      <td>126</td>
      <td>72</td>
      <td>...</td>
      <td>76</td>
      <td>131</td>
      <td>12.5</td>
      <td>2</td>
      <td>0.6</td>
      <td>23</td>
      <td>11</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38981</th>
      <td>40</td>
      <td>170</td>
      <td>105</td>
      <td>124.0</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>1</td>
      <td>1</td>
      <td>141</td>
      <td>85</td>
      <td>...</td>
      <td>48</td>
      <td>138</td>
      <td>17.1</td>
      <td>1</td>
      <td>0.8</td>
      <td>24</td>
      <td>23</td>
      <td>35</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38982</th>
      <td>40</td>
      <td>160</td>
      <td>55</td>
      <td>75.0</td>
      <td>1.5</td>
      <td>1.5</td>
      <td>1</td>
      <td>1</td>
      <td>95</td>
      <td>69</td>
      <td>...</td>
      <td>79</td>
      <td>116</td>
      <td>12.0</td>
      <td>1</td>
      <td>0.6</td>
      <td>24</td>
      <td>20</td>
      <td>17</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>38983</th>
      <td>55</td>
      <td>175</td>
      <td>60</td>
      <td>81.1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
      <td>114</td>
      <td>66</td>
      <td>...</td>
      <td>64</td>
      <td>137</td>
      <td>13.9</td>
      <td>1</td>
      <td>1.0</td>
      <td>18</td>
      <td>12</td>
      <td>16</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>38984 rows × 23 columns</p>
</div>

## EDA
![image](https://github.com/mondayy1/SmokerPrediction/assets/128250130/d9a14a4b-68ca-440e-9400-697d03b53b84)</br>
According to the Plot, height and weight are regarded as important feature.
So I added 'BMI' as a new feature, but it didnt helped.

![image](https://github.com/mondayy1/SmokerPrediction/assets/128250130/fcda0f6a-8459-42cc-aad9-14a91750b1ad)</br>

Categorical features don't help our model at all according to the plot.
So I dropped all these features(Hearing, Urine Protine...)

Scaler with MinMax/Standard didnt help our model according to the test.

## Model
Compared with 5 models.
```python
models = {}
models['Logistic Regression'] = LogisticRegression()
models['Support Vector Machines'] = LinearSVC()
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Gradient Boost'] = GradientBoostingClassifier()
```

## Test Model


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>F1 Score</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Random Forest</th>
      <td>0.721696</td>
      <td>0.796332</td>
      <td>0.718925</td>
      <td>0.724490</td>
    </tr>
    <tr>
      <th>Gradient Boost</th>
      <td>0.661066</td>
      <td>0.746441</td>
      <td>0.673184</td>
      <td>0.649377</td>
    </tr>
    <tr>
      <th>Decision Trees</th>
      <td>0.645094</td>
      <td>0.738361</td>
      <td>0.647346</td>
      <td>0.642857</td>
    </tr>
    <tr>
      <th>Logistic Regression</th>
      <td>0.557666</td>
      <td>0.710273</td>
      <td>0.497207</td>
      <td>0.634864</td>
    </tr>
    <tr>
      <th>Support Vector Machines</th>
      <td>0.420849</td>
      <td>0.692189</td>
      <td>0.304469</td>
      <td>0.681250</td>
    </tr>
  </tbody>
</table>
</div>
Random Forest shows best F1-Score and Auc.


## TODO: GridSearch with Optuna

