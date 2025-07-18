Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at
 Various Battery Specifications and Discharge Cycles
 Kei Long Wong∗, Michael Bosello†, Rita Tse∗, Carlo Falcomer†, Claudio Rossi‡, Giovanni Pau†,♯
 ∗School of Applied Sciences, Macao Polytechnic Institute, Macao SAR
 †DISI-Dept. of Computer Science and Engineering, University of Bologna, Italy
 ‡DEI-Dept. of Electrical, Electronic, and Information Engineering, University of Bologna, Italy
 ♯ UCLAComputer Science, Los Angeles, USA
 ABSTRACT
 Lithium-ion battery technologies play akeyroleintransformingthe
 economy reducing its dependency on fossil fuels. Transportation,
 manufacturing, and services are being electrified. The European
 Commission predicts that in Europe everything that can be electri
f
 ied will be electrified within a decade. The ability to accurate state
 of charge (SOC) estimation is crucial to ensure the safety of the
 operation of battery-powered electric devices and to guide users
 taking behaviors that can extend battery life and re-usability. In
 this paper, we investigate how machine learning models can predict
 the SOC of cylindrical Li-Ion batteries considering a variety of cells
 under different charge-discharge cycles.
 CCSCONCEPTS
 • Computingmethodologies→Neuralnetworks;Supervised
 learning by regression.
 KEYWORDS
 Lithium-ion battery, long short-term memory, recurrent neural
 network, state of charge estimation
 ACMReference Format:
 Kei Long Wong∗, Michael Bosello†, Rita Tse∗, Carlo Falcomer†, Claudio
 Rossi‡, Giovanni Pau†,♯ . 2021. Li-Ion Batteries State-of-Charge Estimation
 Using DeepLSTMatVariousBatterySpecificationsandDischargeCycles.In
 Conference on Information Technology for Social Good (GoodIT ’21), September
 9–11, 2021, Roma, Italy. ACM, New York, NY, USA, 6 pages. https://doi.org/
 10.1145/3462203.3475878
 1 INTRODUCTION
 The transition from fossil fuel to green energy is well known as the
 desired change in our society. To reduce the emission of Carbon
 dioxide (CO2) from conventional transportation, the development
 of Electric Vehicles (EV) is growing quickly. Battery technology
 will be one of the most important key enablers for the green energy
 transition.
 Lithium-ion batteries have been widely used in electric vehicles.
 It is projected that the global EV stock will expand to 140 million by
 Permission to make digital or hard copies of all or part of this work for personal or
 classroom use is granted without fee provided that copies are not made or distributed
 for profit or commercial advantage and that copies bear this notice and the full citation
 on the first page. Copyrights for components of this work owned by others than ACM
 must be honored. Abstracting with credit is permitted. To copy otherwise, or republish,
 to post on servers or to redistribute to lists, requires prior specific permission and/or a
 fee. Request permissions from permissions@acm.org.
 GoodIT ’21, September 9–11, 2021, Roma, Italy
 ©2021 Association for Computing Machinery.
 ACMISBN978-1-4503-8478-0/21/09.
 https://doi.org/10.1145/3462203.3475878
 2030 [1]. Lithium-ion (Li-ion) battery is the most popular adopted
 power supply of EV due to its high energy density, long lifespan,
 lightweight, and low self-discharge rate [19]. Several factors could
 affect the performance and safety of Li-ion battery such as ambient
 temperature, over-charge, or over-discharge [20, 21]. A misuse
 of the battery can lead to a shorter battery life. To overcome these
 issues, Battery Management Systems (BMS) are applied to ensure
 the reliability and stability of the usage of Li-ion batteries.
 One important parameter for the BMS battery health manage
ment is the battery State Of Charge (SOC) estimation which helps
 to prevent the battery from over-charge and over-discharge [10, 28].
 SOC indicates the amount of available charge in the battery which
 can be represented by a value in percentage. This value is intended
 to remain between 0% and 100%, although it is possible to violate
 these limits in an over-discharge or over-charge situation [24]. The
 battery itself does not directly provide information on its SOC value.
 The measurement of SOC value is complex and error-prone due to
 the indirect estimation and the non-linear nature of electrochemical
 reactions in the battery. Relevant information such as the measured
 discharge current, voltage, and ambient temperature can be used
 to measure the SOC indirectly [5].
 Incorrect measurement of SOC could lead to unstable EV per
formance and even shorten the battery life, therefore, reducing
 the environmental benefits of electrification. In general, the SOC
 estimation techniques studied in the literature can be divided into
 three categories: direct methods, model-based methods, and data
driven methods [25]. The direct methods look for the relation
ship between SOC and the physical battery characteristic param
eters. The SOC value can be estimated according to the observed
 parameters[24],[25], [22]. The model-based SOC estimation meth
ods mainlyfocus onmodelingthechemicalandelectrical properties
 of the battery. Commonly, the model-based methods are used in
 collaboration with adaptive filters such as Kalman filter, H-infinity
 f
 ilter, and Particle filter, etc [25], [23]. Model-based methods require
 a comprehensive understanding of the electrochemical properties
 in the battery domain and cannot be used for SOC forecast[12][3].
 This work proposes a data-driven approach for SOC estimation
 based on Deep Learning techniques. Deep learning, which can ap
proximate non-linear functions, is a widely adopted data-driven
 method to tackle the battery SOC estimation problem [27]. Given a
 sufficient amount of training data and an appropriate configuration,
 the SOC value can be predicted accurately without the need for
 a sophisticated electrochemical model. Different types of neural
 networks (NNs) such as Convolutional Neural Networks (CNNs)
 and Recurrent Neural Networks (RNNs) have been studied in the lit
erature to solve various problems of different nature [8, 14]. Among
 85
GoodIT ’21, September 9–11, 2021, Roma, Italy
 Wong, Bosello, Tse, et al.
 them, RNNs are designed to handle sequential data, and they have
 been well studied in the domain of speech recognition and natural
 language processing with successful outcomes [9, 30]. However,
 RNNs struggle to handle long-term dependencies as long time se
ries could cause exploding/vanishing gradient during the training
 phase. To tackle this problem, Hochreiter and Schmidhuber [11]
 proposed the use of RNNs with Long Short-Term Memory (LSTM)
 cells which can correlate a long-range of precedent information.
 In the literature, various applications of LSTM for SOC estima
tion have been proposed. Chemali, Kollmeyer, Preindl, Ahmed, and
 Emadi [4] showcased the ability of LSTM for SOC estimation. They
 used a dataset composed of discharge cycles obtained from a Li-ion
 battery cell with 2.9Ah nominal capacity [17] and collected through
 laboratory testing under different driving profiles. The proposed
 model was validated against various ambient temperatures with
 accurate estimation results achieved. Similarly, Yang, Song, Xu, and
 Tsui [29] proposed the use of a deep LSTM network with data col
lected from an experiment on a 1.1Ah nominal capacity battery
 cell. In [26], a neural network combining CNN and LSTM layers
 was proposed for battery SOC estimation. The research result has
 shown that the CNN part helps to extract the spatial features from
 the input data (voltage, current, and temperature) while the LSTM
 layers explore the correlation of current SOC and historical input
 data. Last but not least, the use of an LSTM encoder-decoder al
gorithm was proposed by Cui, Yong, Kim, Hong, and Joe [7] with
 accurate estimation result against both room temperature and vari
ous temperature conditions. The proposed model was trained and
 tested on 2.0Ah nominal capacity Li-ion battery cell data featuring
 various drive cycles.
 Although the neural network data-driven approaches for SOC
 estimation are widely studied most of the literature mainly focuses
 on datasets containing only one particular battery model or
 setup. This study uses two different Li-ion cell datasets. The first
 oneisoriginal andit hasbeencollectedbytheUniversityofBologna
 (UNIBO), namely the ‘UNIBO Powertools Dataset’. The second one
 is public, the LG 18650HG2 Li-ion battery data [18]. The use of deep
 LSTMnetworks is proposed to perform SOC estimation. Due to the
 heterogeneity of the data collection process and the sampling rate
 of the two datasets, two deep LSTM models with different setups
 are employed in this research. The deep networks are tested on
 Li-ion battery cells with different nominal capacities, specifications,
 and brands; the discharge cycles are produced by both constant
 current discharge and several dynamic driving profiles (such as the
 Urban dynamometer driving schedule (UDDS) [2]).
 The rest of this paper is organized as follows. The employed
 battery datasets are introduced in section 2. Then, section 3 explains
 the proposed deep LSTM models. In section 4, the results of the
 experiments are presented. Finally, section 5 concludes this paper.
 2 LI-IONBATTERYDATASETS
 In this paper, two Li-ion battery datasets– one original and one
 public– with different features are used. The UNIBO Powertools
 Dataset is presented here for the first time, and it is available here1.
 Only the discharge cycles are used in the experiments. The two
 datasets are briefly introduced in the following sections.
 1https://doi.org/10.17632/n6xg5fzsbv.1
 Table 1: UNIBO Powertools Dataset summary
 Test type
 Nominal capacity Cell amount
 Standard
 High current
 Preconditioned
 4.0Ah
 3.0Ah
 2
 4
 2.85Ah
 4
 2.0Ah
 3.0Ah
 2.85Ah
 3.0Ah
 2.1 UNIBOPowertools Dataset
 6
 3
 2
 5
 The UNIBO Powertools Dataset has been collected in a laboratory
 test by an Italian Equipment producer. The cycling experiments
 are designed to analyze different cells intended for use in various
 cleaning equipment such as vacuum and automated floor cleaners.
 The vast dataset is composed of 27 batteries, and it is summarized
 in Table 1. The main features of the dataset are: (1) the use of bat
teries from different manufacturers, (2) cells with several nominal
 capacities, (3) cycling is performed until the cell’s end of life and
 thus data regarding the cell at different life stages are produced,
 which is useful to assess how SOC is affected by the cell’s age
 and State of Health (SOH) as well as to validate the capability of the
 proposed model on estimating SOC under different health status.
 Three types of tests have been conducted. (I) The standard test,
 where the battery was discharged at 5A current in main cycles.
 (II), the high current test, where the battery was discharged at
 8A current in main cycles. (III), the preconditioned test, where
 the battery cells are stored at 45°C environment for 90 days before
 conducting the test.
 During discharge, the sampling period is 10 seconds. The experi
ments were conducted using the following procedure:
 (1) Charge cycle: Constant Current-Constant Voltage (CC-CV)
 at 1.8A and 4.2V (100mA cut-off)
 (2) Dischargecycle: ConstantCurrentuntilcut-offvoltage(2.5V)
 (3) Repeat steps 1 and 2 (main cycle) 100 times
 (4) Capacity measurement: charge CC-CV 1A 4.2V (100mA cut
off) and discharge CC 0.1A 2.5V
 (5) Repeat the above steps until the end of life of the battery cell
 2.2 LG18650HG2Li-ion Battery Data
 The public LG 18650HG2 Li-ion Battery Dataset, published by
 Kollmeyer, Vidal, Naguib, andSkells [18], wasobtainedfromMende
ley data. In the dataset, a series of tests were performed under six
 different temperatures. The battery was charged at 1C rate to 4.2V
 with 50mA cut off before each discharge test. The values measured
 in the discharge cycles are captured at 0.1 seconds sampling period.
 Different drive cycles such as UDDS, LA92, and US06, as well as
 mixes of them, are applied in the discharge tests. In this paper, the
 discharge cycles with temperature of 0°C, 10°C and 25°C were used
 for training and testing the proposed model.
 3 METHODOLOGY
 In this section, the basic theories of RNN and LSTM are introduced
 and the two proposed deep LSTM models are briefly introduced.
 Then, the normalization method used to scale the input data is
 reviewed. Lastly, the model’s configuration is discussed.
 86
Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at Various Battery Specifications and Discharge Cycles
 GoodIT ’21, September 9–11, 2021, Roma, Italy
 3.1 Recurrent Neural Networks Primer
 Recurrent neural networks are a class of neural networks that
 allows the information to persist over time. Different from the feed
forward neural networks that are acyclic directed graphs, RNNs
 have connections within layers forming cyclic directed graphs. This
 empowers neural networks to have a state, and thus memory. The
 information from the previous state is utilized as input along with
 the current time step. It is useful for sequential data prediction as
 relationships between current and past information are considered.
 An example of the architecture of an RNN for SOC estimation
 unfolded in time, is depicted in Fig. 1. The input vector at the time
 step t contains battery parameters such as voltage, current, and
 temperature, and it is denoted as 𝐼𝑛𝑝𝑢𝑡𝑡. ℎ𝑡 represents the hidden
 state at time step t, while the output SOC value at time step t is
 denoted as 𝑆𝑂𝐶𝑡. Fig. 1 demonstrates a common approach for time
series called many-to-many, where multiple input steps are fed to
 the network with one prediction made at each step. Whereas, there
 are other approaches such as the many-to-one and one-to-many,
 where in the first case multiple time-steps are fed with one output
 produced, and in the second case one input is used to produce
 multiple time-steps. As the two battery datasets have very different
 sampling frequencies, we used the many-to-many approach for
 the first model (low-frequency sampling) while in the second one
 (high-frequency sampling) we used the many-to-one approach.
 Figure 1: RNN architecture for SOC estimation unfolded in time
 3.2 LongShort-Term Memory Primer
 The long short-term memory is a type of RNN which is widely
 used to learn long-term dependencies without experiencing the
 exploding and vanishing gradient problems. The forward pass of an
 LSTM cell can be defined by the following steps. In the equations,
 𝑓𝑡 , 𝑖𝑡 , 𝑜𝑡 are the forget-gate, input-gate, output-gate; 𝑐𝑡 and ℎ𝑡 are
 the cell state and hidden state at time step t respectively; 𝜎 is the
 sigmoid function; ⊙ istheHadamardproduct;𝑊 denotestheweight
 matrix; 𝑥𝑡 is the input vector at time step t and 𝑏 is the bias.
 The first step in the LSTM cell is to determine what information
 will be forgotten from the cell state 𝑐𝑡−1. The forget-gate uses a
 sigmoid function, in which outputs are always between 0 and 1.
 Theresult represents therefore how much should be forgotten, with
 0 and 1 representing respectively discarding everything or keeping
 everything from the previous cell’s state. As shown in the equations,
 the decisions of gates are based on the current input and hidden
 state as well as on the network’s weights and biases.
 𝑓𝑡 = 𝜎(𝑊𝑓
 𝑥 𝑥𝑡 +𝑊𝑓
 ℎℎ𝑡−1 +𝑏𝑓 )
 (1)
 The second step determines whether the information will be
 stored in the cell state. There are two parts in the secondstep. Firstly,
 the input-gate with sigmoid output determines to what extent the
 value will be remembered. Secondly, the tanh layer generates the
 new value ˜ 𝑐𝑡 that is multiplied by the sigmoid output and then
 added to the cell state.
 𝑖𝑡 = 𝜎(𝑊𝑖 𝑥𝑥𝑡 +𝑊𝑖
 ℎℎ𝑡−1 +𝑏𝑖)
 ˜
 𝑐𝑡 = 𝑡𝑎𝑛ℎ(𝑊𝑐 𝑥𝑥𝑡 +𝑊𝑐
 ℎℎ𝑡−1 +𝑏𝑐)
 (2)
 The cell state 𝑐𝑡 is then update combining the previous cell state
 𝑐𝑡−1 with new value ˜ 𝑐𝑡 as mentioned above. The forget-gate 𝑓𝑡 and
 input-gate 𝑖𝑡 determine whether the values should be discarded or
 remembered.
 𝑐𝑡 = 𝑓𝑡 ⊙𝑐𝑡−1 +𝑖𝑡 ⊙ ˜ 𝑐𝑡
 (3)
 In thelast step, the output-gate with the sigmoidfunctiondecides
 which part of the cell state is propagated to the hidden state ℎ𝑡. In
 the hidden state, the cell state 𝑐𝑡 is passed via 𝑡𝑎𝑛ℎ and multiplied
 by the output-gate to keep only the desired output.
 𝑜𝑡 = 𝜎(𝑊𝑜 𝑥𝑥𝑡 +𝑊𝑜
 ℎℎ𝑡−1 +𝑏𝑜)
 ℎ𝑡 =𝑜𝑡 ⊙𝑡𝑎𝑛ℎ(𝑐𝑡)
 3.3 Proposed LSTM Approach
 (4)
 There are two deep LSTM models proposed in this paper, one for
 each dataset, as they have very different cycle lengths. Scaled ex
ponential linear units (SELU) [16] activation function is used in all
 the LSTM cells and hidden dense layers. In the output layer, the
 linear activation function is applied to produce the final SOC value.
 The first model is used for the UNIBO dataset. It is a deep neural
 network with three LSTM layers followed by two dense layers to
 map the learned states to desired SOC output. The number of cells
 of each LSTMlayeris256,256,and128respectively. Fig. 2 illustrates
 the architecture of the first proposed model. The first layer is the
 input layer with battery parameters including voltage 𝑉, current 𝐼,
 and temperature𝑇 at each step 𝑡. Since it is a deep LSTM network,
 each LSTM layer returns a sequence which means that each step
 is propagated to the next layer. Here, we adopted the many-to
many approach, the SOC value is therefore estimated at every step.
 The input time series fed to the deep LSTM network is defined as
 [𝐼𝑛𝑝𝑢𝑡𝑡0, 𝐼𝑛𝑝𝑢𝑡𝑡1, ...𝐼𝑛𝑝𝑢𝑡𝑡𝑛], where 𝑛 is the number of steps in the
 entire discharge cycle, and 𝐼𝑛𝑝𝑢𝑡 = [𝑉𝑡,𝐼𝑡,𝑇𝑡] represents voltage,
 current and temperature at each time step respectively. Although
 the entire discharge cycle is fed to the network, only the part that
 precedes the step under examination is available as input for SOC
 estimation, i.e., the hidden state from previous steps 𝑡 − 1 and the
 current input at step 𝑡 are used to estimate the output at step 𝑡.
 Figure 2: Architecture of the first model
 87
GoodIT ’21, September 9–11, 2021, Roma, Italy
 Wong, Bosello, Tse, et al.
 The second model is used for the LG 18650HG2 Li-ion battery
 dataset. The model is composed of two LSTM layers followed by
 three dense layers. The number of cells of both LSTM layers is 256.
 Fig. 3 shows the architecture of the second proposed model. Since
 the second dataset contains more steps in one discharge cycle due
 to its higher sampling rate (0.1 seconds sampling time), the many
to-one approach is more appropriate. In this case, for each 𝑛 step as
 input, one output is returned. In the implementation, we used 300,
 500, and 700 as the number of steps. For example, given input steps
 [𝐼𝑛𝑝𝑢𝑡𝑡0, 𝐼𝑛𝑝𝑢𝑡𝑡1, ...𝐼𝑛𝑝𝑢𝑡𝑡500], the model should estimate the SOC
 value at step 500.
 Figure 3: Architecture of the second model
 3.4 DataNormalization
 Since the input features have different ranges, such as the tem
perature has much higher values than voltage and current, the
 trained model could give more importance to this feature over the
 others due to its larger value. To avoid this problem, the minimum
maximum normalization method is used to scale all input features
 into the same common scale.
 3.5 ModelTraining
 The proposed models are implemented by using the Keras library
 [6]. The Adam algorithm [15] is chosen as the optimizer to update
 the network weights and biases with the learning rate configured
 as 0.00001. All proposed models are trained for 1000 epochs, but
 the training process would stop earlier if there is no further im
provement of validation loss within 50 epochs. The Huber loss [13]
 is used as the loss function. Its peculiarity is that it can be quadratic
 or linear depending on the error value.
 4 RESULTSANDDISCUSSION
 The proposed deep LSTM models are trained and tested using the
 two aforementioned datasets. The model performance against each
 dataset is discussed in this section. The source code of the model
 implementation and results are available here2.
 Root Mean Square Error (RMSE) and Mean absolute error (MAE)
 are used to evaluate the proposed models. The Mean Square Error
 (MSE) is the sum of squared distances between the target and pre
dicted variables divided by the number of samples. The RMSE is
 the square root of the MSE which scales the output value to the
 same scale as MAE. It is more sensitive to outliers as it penalizes the
 model by squaring the error. The MAE on the other hand is more
 robust to outliers as the error is not squared. MAE is an L1 loss
 2https://github.com/KeiLongW/battery-state-estimation
 Table 2: UNIBO dataset tests performance
 Test type
 Nominal capacity MAE RMSE
 Standard
 High current
 Preconditioned
 4.0Ah
 3.0Ah
 2.68% 3.42%
 0.52% 0.73%
 2.85Ah
 0.31% 0.39%
 2.0Ah
 3.0Ah
 2.85Ah
 3.0Ah
 0.59% 0.80%
 0.46% 0.61%
 2.13% 3.24%
 0.47% 0.66%
 function that calculates the sum of the absolute difference between
 the target and predicted variables. The MAE is more suitable for
 problems where the training data present outliers.
 4.1 UNIBOPowertools Dataset
 In the UNIBO dataset tests, the performance of the proposed model
 is evaluated over constant current discharge. The proposed model
 for this dataset was trained with a total of 7738 discharging cycles
 as the training set. One cell for each group of test types (standard,
 high current, pre-conditioned) and nominal capacity was extracted
 as testing data for evaluation purposes. The testing data is isolated
 from training data so that it is unseen during the training process.
 The overall MAE and RMSE on all testing data are 0.69% and 1.34%
 respectively.
 To further investigate the performance of the proposed model,
 Table 2 shows the performance of each test type. The evaluation of
 standard test type with 4.0Ah nominal capacity and high current
 test type with 2.85Ah nominal capacity has the worst performance.
 This is expected as the dataset contains only two cell tests of the
 kind, resulting in one cell used for training and one for testing.
 Whereas, in the other test types with sufficient data the model can
 achieve accurate results with RMSE lower than 1%.
 The examples of SOC estimation results of the proposed model
 on the standard, high current, and preconditioned test types are
 shown in Fig. 4, Fig. 5, and Fig. 6 respectively. The first and the
 last discharge cycles within the entire test of each battery cell are
 presented to demonstrate the SOC estimation performance under
 different health statuses. All results show the discharge process
 of SOC being discharged from 100% to 0%. The x-axis represents
 the discharge steps over the whole discharging cycle and the y
axis represents the SOC value at each step. The black line is the
 actual observed SOC value during the discharge process and the
 red dashed line is the SOC value estimated by the proposed model.
 The model estimates the SOC of the 3.0Ah nominal capacity
 cells correctly and without large fluctuation in each of the three
 test types. Furthermore, the estimations of standard test types with
 2.0Ah and 2.85Ah nominal capacity are accurate too. SOC in both
 the first and last cycle are estimated accurately which suggests that
 the proposed model is capable to estimate SOC under different bat
tery health statuses. In addition, good performance is achieved from
 the preconditioned test type which demonstrates that the storage
 temperature before testing does not affect the battery discharging
 behavior significantly in terms of SOC estimation. On the other
 hand, there are some errors during the ending steps of standard
 4.0Ah nominal capacity and high current 2.85Ah nominal capacity
 battery cell cycles. It is acceptable as there is only one training
 example of that kind of setup.
 88
Li-IonBatteriesState-of-ChargeEstimationUsingDeepLSTMatVariousBatterySpecificationsandDischargeCycles GoodIT’21,September9–11,2021,Roma,Italy
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (a) Standard2.0Ah(firstcycle)
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (b) Standard2.0Ah(lastcycle)
 0 1000 2000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (c) Standard2.85Ah(firstcycle)
 0 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (d) Standard2.85Ah(lastcycle)
 0 1000 2000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (e) Standard3.0Ah(firstcycle)
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (f) Standard3.0Ah(lastcycle)
 0 1000 2000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (g) Standard4.0Ah(firstcycle)
 0 1000 2000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (h) Standard4.0Ah(lastcycle)
 Figure4:UNIBOdatasetSOCestimationresults(standard)
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (a)Highcurrent2.85Ah(firstcycle)
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (b)Highcurrent2.85Ah(lastcycle)
 0 500 1000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (c)Highcurrent3.0Ah(firstcycle)
 0 250 500 Time(s)
 0
 1
 SOC
 Actual Estimated
 (d)Highcurrent3.0Ah(lastcycle)
 Figure5:UNIBOdatasetSOCestimationresults(highcurrent)
 0 1000 2000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (a) Preconditioned3.0Ah(firstcycle)
 0 500 Time(s)
 0
 1
 SOC
 Actual Estimated
 (b) Preconditioned3.0Ah(lastcycle)
 Figure6:UNIBOdatasetSOCestimationresults(preconditioned)
 Table3:LG18650HG2datatestsperformance
 Temp.(°C) 300Steps 500Steps 700Steps
 MAE RMSE MAE RMSE MAE RMSE
 0 1.69% 2.27% 1.47% 2.23% 1.65% 2.60%
 10 1.61% 2.12% 1.57% 2.12% 2.22% 2.89%
 25 1.17% 1.57% 1.59% 2.02% 1.92% 2.64%
 4.2 LG18650HG2Li-ionBatteryData
 IntheLG18650HG2Li-ionbatterydataset,theperformanceofthe
 proposedmodelunderdynamicdischargecurrentisevaluated.Six
 mixeddrivingcyclesforeachofthreedifferenttemperatures0°C,
 10°C,and25°Cwereusedastrainingset.Wehavealsotestedthree
 differenttimeserieslengths,withanumberofstepsof300,500,and
 700,whichareapproximatelyequalto30seconds,50seconds,and
 70secondsdepthintimerespectively.Thetestsetwascomposedof
 aUDDS,anLA92,andaUS06drivingcycleplusonemixeddriving
 cycleforeachofthethreedifferenttemperaturesavailableinthe
 dataset.
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (a) 25°C(300steps)
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (b) 25°C(500steps)
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (c) 25°C(700steps)
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (d) 10°C(300steps)
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (e) 10°C(500steps)
 0 2500 5000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (f) 10°C(700steps)
 0 2000 4000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (g) 0°C(300steps)
 0 2000 4000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (h) 0°C(500steps)
 0 2000 4000 Time(s)
 0
 1
 SOC
 Actual Estimated
 (i) 0°C(700steps)
 Figure7:LG18650HGdataSOCestimationresults(mixedcycles)
 89
GoodIT ’21, September 9–11, 2021, Roma, Italy
 Wong, Bosello, Tse, et al.
 The MAEandRMSEachieved by the 300 steps model are 1.47%
 and 1.99%. The 500 steps one reached an MAE and RMSE of 1.54%
 and 2.12%. The 700 steps model achieved 1.94% MAE and 2.72%
 RMSE. All the aforementioned results were tested with testing data
 under all temperatures. The model performance under each tem
perature with different input lengths is listed in Table 3. Among
 all the configurations, the best performance is achieved from test
ing data under 25°C temperature with 300 steps in input, which
 demonstrates that the battery operates most stably under room
 temperature. The model is able to learn the battery behavior under
 room temperature through the provided driving cycles without
 the need for a long history. While, under 10°C and 0°C tempera
tures, better performance is gained from the 500 input model. This
 indicates that increasing input steps could help to improve the
 estimation result under temperatures that are lower than room
 temperature. However, the worst results are from the 700 input
 steps which suggest that the increment of input steps must be se
lected carefully for the many-to-one approach as an inappropriate
 increment of input steps could result in performance degradation.
 The SOC estimation results on the mixed driving cycles under 0°C,
 10°C and 25°C temperatures are displayed in Fig. 7. The estimation
 results under the three temperatures are competitive and without
 significant errors. Still, errors can be seen from the ending steps
 in mixed cycles under 0°C temperature due to their more dynamic
 discharge pattern.
 5 FINALREMARKS
 In this paper, a deep LSTM NN is proposed to estimate SOC over
 two different Li-ion battery datasets. Discharge cycles with both
 constant and dynamic current under various ambient temperatures
 are used to train and test the proposed models. The evaluation
 results show that the proposed models can learn the battery dy
namic behavior during discharge. Battery SOC can be estimated
 accurately by using the measured voltage, current, and temperature
 values, with 1.34% and1.99%RMSEinconstantcurrentanddynamic
 current discharge cycle respectively. We have also shown how the
 proposedestimation is robust w.r.t. different State of Health statuses.
 The SOH is another important parameter for battery management.
 As future work, we suggest using deep LSTM networks for SOH
 estimation as we believe it can be effective as well.
 REFERENCES
 [1] International Energy Agency. 2020. Global EV Outlook 2020. OECD Publishing,
 Paris. 276 pages. https://doi.org/10.1787/d394399e-en
 [2] United States Environmental Protection Agency. 2020. EPA Urban Dynamometer
 Driving Schedule (UDDS). https://www.epa.gov/emission-standards-reference
guide/epa-urban-dynamometer-driving-schedule-udds
 [3] Christian Campestrini, Thomas Heil, Stephan Kosch, and Andreas Jossen. 2016.
 A comparative study and review of different Kalman filters by applying an
 enhanced validation method. Journal of Energy Storage 8 (2016), 142–159. https:
 //doi.org/10.1016/j.est.2016.10.004
 [4] Ephrem Chemali, Phillip J. Kollmeyer, Matthias Preindl, Ryan Ahmed, and Ali
 Emadi. 2018. Long Short-Term Memory Networks for Accurate State-of-Charge
 Estimation of Li-ion Batteries. IEEE Transactions on Industrial Electronics 65, 8
 (2018), 6730–6739. https://doi.org/10.1109/TIE.2017.2787586
 [5] K.W.E.Cheng,B.P.Divakar,HongjieWu,KaiDing,andHoFaiHo.2011. Battery
Management System (BMS) and SOC Development for Electrical Vehicles. IEEE
 Transactions on Vehicular Technology 60, 1 (2011), 76–88. https://doi.org/10.1109/
 TVT.2010.2089647
 [6] François Chollet and Others. 2015. Keras. https://keras.io.
 [7] Shengmin Cui, Xiaowa Yong, Sanghwan Kim, Seokjoon Hong, and Inwhee Joe.
 2020. An LSTM-Based Encoder-Decoder Model for State-of-Charge Estimation
 of Lithium-Ion Batteries. In Intelligent Algorithms in Software Engineering, Radek
 Silhavy (Ed.). Springer International Publishing, Cham, 178–188.
 [8] Wim DeMulder, Steven Bethard, and Marie-Francine Moens. 2015. A survey on
 the application of recurrent neural networks to statistical language modeling.
 Computer Speech and Language 30, 1 (2015), 61–98. https://doi.org/10.1016/j.csl.
 2014.09.005
 [9] Alex Graves, Abdel-rahman Mohamed, and Geoffrey Hinton. 2013. Speech
 recognition with deep recurrent neural networks. In 2013 IEEE International
 Conference on Acoustics, Speech and Signal Processing. IEEE, Piscataway, 6645
6649. https://doi.org/10.1109/ICASSP.2013.6638947
 [10] M.A. Hannan, M.S.H. Lipu, A. Hussain, and A. Mohamed. 2017. A review of
 lithium-ion battery state of charge estimation and management system in electric
 vehicle applications: Challenges and recommendations. Renewable and Sustain
able Energy Reviews 78 (2017), 834–854. https://doi.org/10.1016/j.rser.2017.05.001
 [11] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long Short-Term Memory.
 Neural Computation 9, 8 (1997), 1735–1780. https://doi.org/10.1162/neco.1997.9.
 8.1735
 [12] Dickson N. T. How, M. A. Hannan, M. S. Hossain Lipu, and Pin Jern Ker. 2019.
 State of Charge Estimation for Lithium-Ion Batteries Using Model-Based and
 Data-Driven Methods: A Review. IEEE Access 7 (2019), 136116–136136. https:
 //doi.org/10.1109/ACCESS.2019.2942213
 [13] Peter J. Huber. 1992. Robust Estimation of a Location Parameter. Springer New
 York, New York, NY, 492–518. https://doi.org/10.1007/978-1-4612-4380-9_35
 [14] Asifullah Khan, Anabia Sohail, Umme Zahoora, and Aqsa Saeed Qureshi. 2020.
 A survey of the recent architectures of deep convolutional neural networks.
 Artificial Intelligence Review 53, 8 (2020), 5455–5516.
 [15] DiederikP.KingmaandJimmyBa.2015. Adam:AMethodforStochasticOptimiza
tion. In 3rd International Conference on Learning Representations, Yoshua Bengio
 and Yann LeCun (Eds.). ICLR, San Diego, 1–15. http://arxiv.org/abs/1412.6980
 [16] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, and Sepp Hochreiter.
 2017. Self-Normalizing Neural Networks. In Proceedings of the 31st International
 Conference on Neural Information Processing Systems (Long Beach, California,
 USA) (NIPS’17). Curran Associates Inc., Red Hook, NY, USA, 972–981.
 [17] Phillip Kollmeyer. 2018. Panasonic 18650PF Li-ion Battery Data. https://doi.org/
 10.17632/wykht8y7tg.1
 [18] Phillip Kollmeyer, Carlos Vidal, Mina Naguib, and Michael Skells. 2020. LG
 18650HG2 Li-ion Battery Data and Example Deep Neural Network xEV SOC
 Estimator Script. https://doi.org/10.17632/cp3473x7xv.3
 [19] Reiner Korthauer. 2018. Lithium-ion batteries: basics and applications. Springer,
 Berlin.
 [20] Shuai Ma, Modi Jiang, Peng Tao, Chengyi Song, Jianbo Wu, Jun Wang, Tao Deng,
 and Wen Shang. 2018. Temperature effect and thermal impact in lithium-ion
 batteries: A review. Progress in Natural Science: Materials International 28, 6
 (2018), 653–666. https://doi.org/10.1016/j.pnsc.2018.11.002
 [21] Dongxu Ouyang, Mingyi Chen, Jiahao Liu, Ruichao Wei, Jingwen Weng, and
 Jian Wang. 2018. Investigation of a commercial lithium-ion battery under
 overcharge/over-discharge failure conditions. RSC advances 8, 58 (2018), 33414
33424.
 [22] GianfrancoPistoia. 2014. Lithium-ion batteries: advances and applications. Elsevier
 Science, Amsterdam.
 [23] Gregory L. Plett. 2004. Extended Kalman filtering for battery management sys
tems of LiPB-based HEV battery packs: Part 2. Modeling and identification. Jour
nal of Power Sources 134, 2 (2004), 262–276. https://doi.org/10.1016/j.jpowsour.
 2004.02.032
 [24] Gregory L Plett. 2015. Battery management systems, Volume II: Equivalent-circuit
 methods. Artech House, United States.
 [25] Juan Pablo Rivera-Barrera, Nicolás Muñoz-Galeano, and Henry Omar Sarmiento
Maldonado. 2017. SoC estimation for lithium-ion batteries: Review and future
 challenges. Electronics 6, 4 (2017), 102.
 [26] Xiangbao Song, Fangfang Yang, Dong Wang, and Kwok-Leung Tsui. 2019. Com
bined CNN-LSTM Network for State-of-Charge Estimation of Lithium-Ion Bat
teries. IEEE Access 7 (2019), 88894–88902. https://doi.org/10.1109/ACCESS.2019.
 2926517
 [27] Carlos Vidal, Pawel Malysz, Phillip Kollmeyer, and Ali Emadi. 2020. Machine
 Learning Applied to Electrified Vehicle Battery State of Charge and State of
 Health Estimation: State-of-the-Art. IEEE Access 8 (2020), 52796–52814. https:
 //doi.org/10.1109/ACCESS.2020.2980961
 [28] Yidan Xu, Minghui Hu, Anjian Zhou, Yunxiao Li, Shuxian Li, Chunyun Fu, and
 Changchao Gong. 2020. State of charge estimation for lithium-ion batteries
 based on adaptive dual Kalman filter. Applied Mathematical Modelling 77 (2020),
 1255–1272. https://doi.org/10.1016/j.apm.2019.09.011
 [29] Fangfang Yang, Xiangbao Song, Fan Xu, and Kwok-Leung Tsui. 2019. State
of-Charge Estimation of Lithium-Ion Batteries via Long Short-Term Memory
 Network. IEEE Access 7 (2019), 53792–53799. https://doi.org/10.1109/ACCESS.
 2019.2912803
 [30] Wenpeng Yin, Katharina Kann, Mo Yu, and Hinrich Schütze. 2017. Compara
tive study of CNN and RNN for natural language processing. arXiv preprint
 abs/1702.01923 (2017).
 90