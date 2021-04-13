## Loading modules
from Data import Data
from System import System, OdeSettings, LossSettings, RateSettings
from HybridModel import HybridModel
from TimeSeriesPair import TimeSeriesPair
from mpl_toolkits import mplot3d
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
hyperparameter_list1 = [1,2,3,4,5]
hyperparameter_list2 = [0]

iteration = 0

hyperparameters_results = pd.read_excel("Model_hyperparameters_analysis.xlsx")
model_score=[]
model_train_loss=[]
model_val_loss=[]
test_evaluation=[]
for a in range(len(hyperparameter_list1)):
    for b in range(len(hyperparameter_list2)):

        iteration = iteration + 1
        # Set ode and loss settings
        ode_settings = OdeSettings(variable_stepsize=True, time_steps=6, rel_tol=1e-3, abs_tol=1e-6)
        loss_settings = LossSettings(geometry='Sphere', loss_type='Number')
        rate_settings = RateSettings(layer_activations=["elu","softplus"],
                                     layer_neurons=[10,23])

        # Define model system
        system = System(case="FIX_Fermentation_v2", ode_settings=ode_settings,
                        loss_settings=loss_settings, rate_settings=rate_settings, dilution=False,
                        regularization=1, normalize=True)

        # Create data-set and set up data-shuffler
        #df = pd.read_excel("Pivot_data_raw_original.xlsx")

        data = Data(case_id='FIX_fermentation')
        included_variables = ["Biomass","Glucose","Glutamine","Lactate","Glutamate","Ammonium",
                      "FIX_Tank","Osmolality","Offline_pH","Po2"]
        excluded_variables = ["VCD","Viability","K+","Na+","FIX_harvest","Average_cell_size","Pco2"]
        units = [" (10^6 cells/ml)"," (mmol/L)"," (mmol/L)"," (mmol/L)"," (mmol/L)"," (mmol/L)"," (mg/L)"," (mOsm/Kg)",
                 " "," (%)"]
        stats,mean,stdev,range_var = data.load_from_excel('RAW_DATA.xlsx',excluded_variables=excluded_variables,included_variables = included_variables)
        time_series_pair = TimeSeriesPair(data=data, system=system)


        # Split training and validation data
        training_batches = [32,44,45,48,42,43]
        validation_batches = [47]
        test_batches = [41]
        data.set_batch_pool(pool_batch_id=training_batches, pool_type='Training')
        data.set_batch_pool(pool_batch_id=validation_batches, pool_type='Validation')
        data.set_batch_pool(pool_batch_id=test_batches, pool_type='Test')
        # training_df = df[df.Batch.isin(training_batches)]
        # training_df["Type"] = "Training"
        # validation_df = df[df.Batch.isin(validation_batches)]
        # validation_df["Type"] = "Validation"
        # test_df = df[df.Batch.isin(test_batches)]
        # test_df["Type"] = "Test"
        #44 as normal, 32 as abnormal

        # Generate shuffled training and evaluation data
        training_data = time_series_pair.shuffle(pool_type=['Training'], min_step=1, max_step=1)
        validation_data = time_series_pair.shuffle(pool_type=['Validation'], min_step=1, max_step=1)
        test_data = time_series_pair.shuffle(pool_type=['Test'], min_step=1, max_step=1)
        # Set up hybrid training model
        hybrid_model = HybridModel(system=system)
        #Create training and validation data
        x_train, y_train = hybrid_model.model_data(training_data,included_variables)
        x_val, y_val = hybrid_model.model_data(validation_data,included_variables)
        x_test, y_test = hybrid_model.model_data(test_data,included_variables)
        range_inputs = [i for i in range(len(included_variables))]
        for i in range(len(range_inputs)):
            range_inputs[i] = float(stats[included_variables[i]]['max']-stats[included_variables[i]]['min'])
        loss_weights = [max(range_inputs)/range_inputs[i] for i in range(len(range_inputs)-2)]




        # Compile hybrid model
        hybrid_model.training_model.compile(loss=hybrid_model.loss_model.loss,
                                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                                            run_eagerly=False
                                           ,loss_weights = loss_weights
                                            )
        #
        #
        # # Early stopping
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        #
        #
        #Train the model
        batch_size = 30
        epochs = 2000
        hybrid_model.training_history = hybrid_model.training_model.fit(x=x_train, y=y_train,
                                                                        validation_data=(x_val, y_val),
                                                                        callbacks=[es],
                                                                        epochs=epochs,
                                                                        batch_size = batch_size,
                                                                        verbose=2).history
        epochs = len(hybrid_model.training_history['loss'])
        #Evaluating model performance respect a reference (constant) model
        x_train_conc = np.concatenate(x_train[1], axis = -1)
        x_val_conc = np.concatenate(x_val[1], axis = -1)
        x_test_conc = np.concatenate(x_test[1],axis = -1)

        loss_training_reference = hybrid_model.calculate_reference_loss(x_train_conc,y_train,category = 'train')
        loss_validation_reference = hybrid_model.calculate_reference_loss(x_val_conc,y_val,category = 'val')
        loss_test_reference = hybrid_model.calculate_reference_loss(x_test_conc,y_test,category = 'val')
        loss_training_hybrid = min(hybrid_model.training_history['loss'])
        loss_validation_hybrid = min(hybrid_model.training_history['val_loss'])

        # valerror = [(min(hybrid_model.training_history['val_functional_3_loss']))]
        # for i in range(1,len(included_variables)):
        #     valerror.append(min(hybrid_model.training_history['val_functional_3_'+str(i)+'_loss']))
        #
        # valreferenceerror = [2.9868,1.7418,0.4644,2.0326,0.2213,0.2148,1.8292,6.7998,1,1]
        #
        # trainingevalresult = []
        #
        # for i in range(len(valerror)):
        #     trainingevalresult.append((1-(valerror[i] / valreferenceerror[i])) * 100)

        if loss_training_hybrid > loss_training_reference or loss_validation_hybrid > loss_validation_reference:
            print("The model is not performing well")
        else:
            print("The model is working!!!")

        # Loss error and validation error
        loss_error = hybrid_model.training_history['loss']
        validation_error = hybrid_model.training_history['val_loss']
        epoch_range = [i for i in range(epochs)]

        plt.plot(epoch_range,loss_error,label = "Training Loss error")
        plt.plot(epoch_range,validation_error,label = "Validation error")
        # plt.axhline(y=loss_training_reference, label ="Reference Loss error",color='b', linestyle='-')
        # plt.axhline(y=loss_validation_reference, label="Reference Validation error", color='r', linestyle='-')
        plt.title("Loss evaluation")
        plt.xlabel('Number of epochs')
        plt.ylabel('Average sumed absolute error')
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.savefig("lossvsepochs.eps",format = "eps")
        plt.show()

        loss_error = min(hybrid_model.training_history['loss'])
        validation_error = min(hybrid_model.training_history['val_loss'])




        # Single time-step Predictions (1-day and 4-day)########################################################
        single_step_evaluation = hybrid_model.training_model.evaluate(x_test,y_test)
        single_step_prediction = hybrid_model.training_model.predict(x=x_test)
        single_step_prediction_rates = hybrid_model.sub_models['rate'].predict(x=x_test[1])
        single_step_results = [[] for i in range(len(x_test[1]))]
        single_step_results_rates = [[] for i in range(len(hybrid_model.variable_details['y']))]


        first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
        first_values = [[float(first_conditions[i])] for i in range(len(first_conditions))]
        real_value = [test_data['Measured variables'][:,i] for i in range(len(x_test[1]))]
        for i in range(len(x_test[1])):
            real_value[i] = np.append(real_value[i],test_data['Future variables'][26,i])




        #Multi time-step predictions
        first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
        multi_step_results = [[] for i in range(len(x_test[1]))]
        multi_step_results_rates = [[] for i in range(len(hybrid_model.variable_details['y']))]
        times = np.insert(np.cumsum(x_test[2]), 0, 0, axis=0)
        times_rates = np.cumsum(x_test[2])



        for t in range(27):

            measured_constants = [x_test[0][i][t] for i in range(len(x_test[0]))]
            time = np.array([x_test[2][t]])

            result_rates = hybrid_model.sub_models['rate'].predict(x=[first_conditions])
            result = hybrid_model.sub_models['ODE'].predict(x=[result_rates,measured_constants,first_conditions,time])
            print(t,result[6])
            for i in range(2):
                result[i+8] = np.array(real_value[i+8][t+1]).reshape((1,1))

            for i in range(len(result_rates)):
                multi_step_results_rates[i].append(float(result_rates[i]))

            if t==26:


                result[0] = np.array(real_value[0][t]).reshape((1,1))
                result[1] = np.array(real_value[1][t]).reshape((1,1))
                result[2] = np.array(real_value[2][t]).reshape((1,1))
                result[3] = np.array(real_value[3][t]).reshape((1,1))
                result[4] = np.array(real_value[4][t]).reshape((1,1))
                result[5] = np.array(real_value[5][t]).reshape((1,1))
                result[6] = np.array(real_value[6][t]).reshape((1,1))

            # elif t==4:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]
            # elif t==7:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]
            # elif t==11:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]
            # elif t==15:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]
            # elif t==19:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]
            # elif t==23:
            #     result =  [np.array(real_value[i][t+1]).reshape((1,1)) for i in range(len(included_variables))]

            #else:


                #result[0] = np.array(real_value[0][t+1]).reshape((1,1))
                #result[1] = np.array(real_value[1][t+1]).reshape((1,1))
                #result[2] = np.array(real_value[2][t+1]).reshape((1,1))
                #result[3] = np.array(real_value[3][t+1]).reshape((1,1))
                #result[4] = np.array(real_value[4][t+1]).reshape((1,1))
                #result[5] = np.array(real_value[5][t+1]).reshape((1,1))
                #result[7] = np.array(real_value[7][t+1]).reshape((1,1))

            first_conditions = result
            for z in range(len(result)):
                multi_step_results[z].append(float(result[z]))
        #
        #
        #
        #Results plotting
        for i in range(len(single_step_results)):
            multi_step_results[i] = first_values[i] + multi_step_results[i]
            single_step_results[i] = [float(first_values[i][0])] + [float(single_step_prediction[i][j]) for j in range(len(single_step_prediction[i]))]

        for i in range(len(multi_step_results_rates)):
            multi_step_results_rates[i]= np.array(multi_step_results_rates[i])
        for i in range(len(single_step_results_rates)):
            single_step_results_rates[i] = np.array(single_step_prediction_rates[i][:,0])

        multi_step_results_rates = pd.DataFrame(multi_step_results_rates)
        single_step_results_rates = pd.DataFrame(single_step_results_rates)
        multi_step_results_rates = pd.DataFrame.transpose(multi_step_results_rates)
        single_step_results_rates = pd.DataFrame.transpose(single_step_results_rates)

        fig,ax = plt.subplots(4,2,sharex = False,sharey = False,figsize = (10,13))
        for i in range(4):
            for j in range(2):
                ax[i,j].plot(times,single_step_results[i * 2+j],label = "Predictions Batch C-0041",lw = 2,marker = 'o')
                ax[i,j].plot(times,real_value[:][i * 2+j],label = "Real values Batch C-0041",lw = 2,marker = 'o')
                ax[i,j].title.set_text(
                    hybrid_model.variable_details['x'][i * 2+j]['id']+" "+" predicted and real values")

                ax[i,j].errorbar(range(0,28),real_value[:][i * 2+j],stdev.iloc[:,i * 2+j],linestyle = 'None',
                                 marker = '^')
                ax[i,j].grid()
                plt.xticks(range(0,28,2))
                ax[i,j].legend(loc = 'best')
                plt.setp(ax[i,j],ylabel = hybrid_model.variable_details['x'][i * 2+j]['id']+units[i * 2+j])

                # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        fig.tight_layout(pad = 3.0)
        plt.setp(ax[:,:],xlabel = 'Time (Days)')

        fig.suptitle("1-Day-single_step_prediction " +" iteration="+ str(a))

        plt.savefig("1-Day-single_step_prediction_{}.eps".format(a),format = "eps")
        plt.show()

        fig,ax = plt.subplots(4,2,sharex = False,sharey = False,figsize = (10,13))
        for i in range(4):
            for j in range(2):
                ax[i,j].plot(times,multi_step_results[i * 2+j],label = "Predictions Batch C-0041",lw = 2,marker = 'o')
                ax[i,j].plot(times,real_value[:][i * 2+j],label = "Real values Batch C-0041",lw = 2,marker = 'o')
                ax[i,j].title.set_text(
                    hybrid_model.variable_details['x'][i * 2+j]['id']+" "+" predicted and real values")

                ax[i,j].errorbar(range(0,28),real_value[:][i * 2+j],stdev.iloc[:,i * 2+j],linestyle = 'None',
                                 marker = '^')
                ax[i,j].grid()
                plt.xticks(range(0,28,2))
                ax[i,j].legend(loc = 'best')
                plt.setp(ax[i,j],ylabel = hybrid_model.variable_details['x'][i * 2+j]['id']+units[i * 2+j])

                # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        fig.tight_layout(pad = 3.0)
        plt.setp(ax[:,:],xlabel = 'Time (Days)')

        fig.suptitle("4-Day_Multi-step_prediction " +" iteration="+ str(a))

        plt.savefig("4-Day_Multi-step_prediction.eps_{}".format(a),format = "eps")
        plt.show()

        # Predicted vs real plotting

        fig,ax = plt.subplots(4,2,sharex = False,sharey = False,figsize = (10,13))
        for i in range(4):
            for j in range(2):
                x = np.sort(real_value[:][i * 2+j])
                y = np.sort(multi_step_results[i * 2+j])
                m,n = np.polyfit(x,y,1)
                ax[i,j].plot(x,y,label = "Slope = "+str(round(m,3)),lw = 2,marker = 'o')
                ax[i,j].plot(x,m * x+n)
                ax[i,j].title.set_text("Predicted vs measured: "+hybrid_model.variable_details['x'][i * 2+j]['id'])

                ax[i,j].grid()

                ax[i,j].legend(loc = 'best')

                # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        fig.tight_layout(pad = 3.0)
        plt.setp(ax[:,:],xlabel = 'Real values')
        plt.setp(ax[:,:],ylabel = 'Predicted values')

        fig.suptitle("4-Day_Predicted_Real "+" iteration="+str(a))

        plt.savefig("4-Day_Predicted_Real_{}.eps".format(a),format = "eps")
        plt.show()

        #Predictions (27 days)#############################################################
        single_step_prediction = hybrid_model.training_model.predict(x = x_test)
        single_step_results = [[] for i in range(len(x_test[1]))]

        first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
        first_values = [[float(first_conditions[i])] for i in range(len(first_conditions))]
        real_value = [test_data['Measured variables'][:,i] for i in range(len(x_test[1]))]
        for i in range(len(x_test[1])):
            real_value[i] = np.append(real_value[i],test_data['Future variables'][26,i])

        # Multi time-step predictions
        first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
        multi_step_results = [[] for i in range(len(x_test[1]))]
        multi_step_results_rates = [[] for i in range(len(hybrid_model.variable_details['y']))]
        times = np.insert(np.cumsum(x_test[2]),0,0,axis = 0)
        times_rates = np.cumsum(x_test[2])


        for t in range(27):

            measured_constants = [x_test[0][i][t] for i in range(len(x_test[0]))]
            time = np.array([x_test[2][t]])

            result_rates = hybrid_model.sub_models['rate'].predict(x = [first_conditions])
            result = hybrid_model.sub_models['ODE'].predict(x = [result_rates,measured_constants,first_conditions,time])
            print(t,result[6])
            for i in range(2):
                result[i+8] = np.array(real_value[i+8][t+1]).reshape((1,1))

            for i in range(len(result_rates)):
                multi_step_results_rates[i].append(float(result_rates[i]))



            first_conditions = result
            for z in range(len(result)):
                multi_step_results[z].append(float(result[z]))
        #
        #
        #
        # Results plotting
        for i in range(len(single_step_results)):
            multi_step_results[i] = first_values[i]+multi_step_results[i]
            single_step_results[i] = [float(first_values[i][0])]+[float(single_step_prediction[i][j]) for j in
                                                                  range(len(single_step_prediction[i]))]

        for i in range(len(multi_step_results_rates)):
            multi_step_results_rates[i] = np.array(multi_step_results_rates[i])

        multi_step_results_rates = pd.DataFrame(multi_step_results_rates)
        multi_step_results_rates = pd.DataFrame.transpose(multi_step_results_rates)

        fig,ax = plt.subplots(4,2,sharex = False,sharey = False,figsize = (10,13))
        for i in range(4):
            for j in range(2):
                ax[i,j].plot(times,multi_step_results[i * 2+j],label = "Predictions Batch C-0041",lw = 2,
                             marker = 'o')
                ax[i,j].plot(times,real_value[:][i * 2+j],label = "Real values Batch C-0041",lw = 2,marker = 'o')
                ax[i,j].title.set_text(
                    hybrid_model.variable_details['x'][i * 2+j]['id']+" "+" predicted and real values")

                ax[i,j].errorbar(range(0,28),real_value[:][i * 2+j],stdev.iloc[:,i * 2+j],linestyle = 'None',
                                 marker = '^')
                ax[i,j].grid()
                plt.xticks(range(0,28,2))
                ax[i,j].legend(loc = 'best')
                plt.setp(ax[i,j],ylabel = hybrid_model.variable_details['x'][i * 2+j]['id']+units[i * 2+j])

                # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        fig.tight_layout(pad = 3.0)
        plt.setp(ax[:,:],xlabel = 'Time (Days)')

        fig.suptitle("27-Day_Multi-step_prediction "+" iteration="+str(a))

        plt.savefig("27-Day_Multi-step_prediction_{}.eps".format(a),format = "eps")
        plt.show()

        # Predicted vs real plotting

        fig,ax = plt.subplots(4,2,sharex = False,sharey = False,figsize = (10,13))
        for i in range(4):
            for j in range(2):
                x = np.sort(real_value[:][i * 2+j])
                y = np.sort(multi_step_results[i * 2+j])
                m,n = np.polyfit(x,y,1)
                ax[i,j].plot(x,y,label = "Slope = "+str(round(m,3)),lw = 2,marker = 'o')
                ax[i,j].plot(x,m * x+n)
                ax[i,j].title.set_text("Predicted vs measured: "+hybrid_model.variable_details['x'][i * 2+j]['id'])

                ax[i,j].grid()

                ax[i,j].legend(loc = 'best')

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        fig.tight_layout(pad = 3.0)
        plt.setp(ax[:,:],xlabel = 'Real values')
        plt.setp(ax[:,:],ylabel = 'Predicted values')

        fig.suptitle("27-Day_Predicted_Real "+" iteration="+str(a))

        plt.savefig("27-Day_Predicted_Real_{}.eps".format(a),format = "eps")
        plt.show()

        #Model evaluation

        Long_time_horizon_average_error = sum(abs(multi_step_results[6]-real_value[:][6])) / 27
        print("Long_time_horizon_average_error = "+str(Long_time_horizon_average_error))

        # Sensitivity analysis

        input_sensitivity = np.zeros((len(included_variables) ** 2,len(included_variables)))
        vector = np.linspace(0,1,len(included_variables))
        x_plot = np.zeros((len(included_variables),len(included_variables)))
        for i in range(len(included_variables)):
            input_sensitivity[:,i] = stats.iloc[1][i]

        for j in range(len(included_variables)):

            for i in range(len(vector)):
                input_sensitivity[i+j * len(vector),j] = stats.iloc[3][j]+vector[i] * range_inputs[j]
                x_plot[i,j] = stats.iloc[3][j]+vector[i] * range_inputs[j]

        x_sensitivity = [np.expand_dims(input_sensitivity[:,i],axis =1) for i in range(len(included_variables))]

        qFIX = hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[0] * stats.iloc[1][0] * stats.iloc[1][1] * \
               hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[11] \
               +hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[1] * stats.iloc[1][0] * stats.iloc[1][2] * \
               hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[12] \
               +hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[3] * stats.iloc[1][0] * stats.iloc[1][4] * \
               hybrid_model.sub_models['rate'].predict(x = x_sensitivity)[13]

        qFIX = np.split(qFIX,len(included_variables))

        fig,ax = plt.subplots(5,2,sharex = False,sharey = True,figsize = (10,13))
        for i in range(5):
            for j in range(2):
                ax[i,j].plot(x_plot[:,i * 2+j],qFIX[i * 2+j],label = hybrid_model.variable_details['x'][i * 2+j]['id'],
                             lw = 2,marker = 'o')
                ax[i,j].title.set_text(hybrid_model.variable_details['x'][i * 2+j]['id'])

                ax[i,j].grid()

                ax[i,j].legend(loc = 'best')
                plt.setp(ax[i,j],xlabel = hybrid_model.variable_details['x'][i * 2+j]['id']+units[i * 2+j])

                # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])

        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.2  # the amount of height reserved for white space between subplots
        # plt.subplots_adjust(left = left,bottom = bottom,right = right,top = top,wspace = wspace,hspace = hspace)
        fig.tight_layout(pad = 3.0)

        plt.setp(ax[:,:],ylabel = 'rFIX Value')
        fig.suptitle("Sensitivity-analysis "+"iteration="+str(a))

        plt.savefig("Sensitivity-analysis_{}.eps".format(a),format = "eps")
        plt.show()

        print("Long_time_horizon_average_error= "+
              str(Long_time_horizon_average_error)+","+"loss_training ="+str(loss_training_hybrid)
              +","+"val_loss ="+str(loss_validation_hybrid))
        model_score.append(Long_time_horizon_average_error)
        test_evaluation.append(single_step_evaluation)
        hyperparameters_results["Test loss"] = single_step_evaluation[0]
        hyperparameters_results["Long_time_horizon_average_error"] = Long_time_horizon_average_error
        model_val_loss.append(loss_validation_hybrid)
        hyperparameters_results["Validation Loss"] = loss_validation_hybrid
        model_train_loss.append(loss_training_hybrid)
        hyperparameters_results["Training Loss"] = loss_training_hybrid
        hyperparameters_results.to_excel("Model_hyperparameters_analysis.xlsx")
        print("finish")

#     #____________________________________________________________________________________________________________________________________________________________
#
#         # Define model system
#         system = System(case = "FIX_Fermentation_v2",ode_settings = ode_settings,
#                         loss_settings = loss_settings,rate_settings = rate_settings,dilution = False,
#                         regularization = 1,normalize = True)
#
#         # Create data-set and set up data-shuffler
#         # df = pd.read_excel("Pivot_data_raw_original.xlsx")
#
#         data = Data(case_id = 'FIX_fermentation')
#         included_variables = ["Biomass","Glucose","Glutamine","Lactate","Glutamate","Ammonium",
#                               "FIX_Tank","Osmolality","Offline_pH","Po2"]
#         excluded_variables = ["VCD","Viability","K+","Na+","FIX_harvest","Average_cell_size","Pco2"]
#         units = ["10^6 cells/ml","mM","mM","mM","mM","mM","mg/L","pH units","% of saturation"]
#         stats,mean,stdev,range_var = data.load_from_excel('RAW_DATA_TEST.xlsx',excluded_variables = excluded_variables,
#                                                           included_variables = included_variables)
#         time_series_pair = TimeSeriesPair(data = data,system = system)
#
#         # Split training and validation data
#         training_batches = [32,44,45,48,42,43]
#         validation_batches = [47]
#         test_batches = [49]
#         data.set_batch_pool(pool_batch_id = training_batches,pool_type = 'Training')
#         data.set_batch_pool(pool_batch_id = validation_batches,pool_type = 'Validation')
#         data.set_batch_pool(pool_batch_id = test_batches,pool_type = 'Test')
#         # training_df = df[df.Batch.isin(training_batches)]
#         # training_df["Type"] = "Training"
#         # validation_df = df[df.Batch.isin(validation_batches)]
#         # validation_df["Type"] = "Validation"
#         # test_df = df[df.Batch.isin(test_batches)]
#         # test_df["Type"] = "Test"
#         # 44 as normal, 32 as abnormal
#
#         # Generate shuffled training and evaluation data
#         training_data = time_series_pair.shuffle(pool_type = ['Training'],min_step = 1,max_step = 1)
#         validation_data = time_series_pair.shuffle(pool_type = ['Validation'],min_step = 1,max_step = 1)
#         test_data = time_series_pair.shuffle(pool_type = ['Test'],min_step = 1,max_step = 1)
#         x_train, y_train = hybrid_model.model_data(training_data,included_variables)
#         x_val, y_val = hybrid_model.model_data(validation_data,included_variables)
#         x_test, y_test = hybrid_model.model_data(test_data,included_variables)
#
# #Predictions (27 days)#############################################################
#         first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
#         first_values = [[float(first_conditions[i])] for i in range(len(first_conditions))]
#
#         # Multi time-step predictions
#         multi_step_results = [[] for i in range(len(x_test[1]))]
#         multi_step_results_rates = [[] for i in range(len(hybrid_model.variable_details['y']))]
#         times = np.insert(np.cumsum(x_test[2]),0,0,axis = 0)
#         times_rates = np.cumsum(x_test[2])
#
#
#         for t in range(27):
#
#             measured_constants = [x_test[0][i][t] for i in range(len(x_test[0]))]
#             time = np.array([x_test[2][t]])
#
#             result_rates = hybrid_model.sub_models['rate'].predict(x = [first_conditions])
#             result = hybrid_model.sub_models['ODE'].predict(x = [result_rates,measured_constants,first_conditions,time])
#             print(t,result[6])
#             for i in range(2):
#                 result[i+8] = np.array(real_value[i+8][t+1]).reshape((1,1))
#
#             for i in range(len(result_rates)):
#                 multi_step_results_rates[i].append(float(result_rates[i]))
#
#
#
#             first_conditions = result
#             for z in range(len(result)):
#                 multi_step_results[z].append(float(result[z]))
#         #
#         #
#         #
#         # Results plotting
#         for i in range(len(single_step_results)):
#             multi_step_results[i] = first_values[i]+multi_step_results[i]
#
#
#         for i in range(len(multi_step_results_rates)):
#             multi_step_results_rates[i] = np.array(multi_step_results_rates[i])
#
#         multi_step_results_rates = pd.DataFrame(multi_step_results_rates)
#         multi_step_results_rates = pd.DataFrame.transpose(multi_step_results_rates)
#
#         fig,ax = plt.subplots(3,3,sharex = False,sharey = False,figsize = (15,10))
#         for i in range(3):
#             for j in range(3):
#
#
#                 ax[i,j].plot(times,multi_step_results[i * 3+j],label = "Predictions Batch C-0049",lw = 2,
#                              marker = 'o')
#                 ax[i,j].title.set_text(
#                     hybrid_model.variable_details['x'][i * 3+j]['id']+" "+" predicted and real values")
#
#                 ax[i,j].grid()
#                 plt.xticks(range(0,28,2))
#                 ax[i,j].legend(loc = 'best')
#                 plt.gcf().autofmt_xdate()
#
#                 # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])
#
#         left = 0.125  # the left side of the subplots of the figure
#         right = 0.9  # the right side of the subplots of the figure
#         bottom = 0.1  # the bottom of the subplots of the figure
#         top = 0.9  # the top of the subplots of the figure
#         wspace = 0.2  # the amount of width reserved for blank space between subplots
#         hspace = 0.2  # the amount of height reserved for white space between subplots
#         plt.subplots_adjust(left = left,bottom = bottom,right = right,top = top,wspace = wspace,hspace = hspace)
#         plt.setp(ax[:,:],xlabel = 'Time (Days)')
#         plt.setp(ax[:,:],ylabel = 'Value')
#
#         fig.suptitle("27-Day_Multi-step_prediction "+"a="+str(a)+", b="+str(b))
#
#         plt.savefig("27-Day_Multi-step_prediction.eps",format = "eps")
#         plt.show()
# #
#
#
#     #____________________________________________________________________________________________________________________________________________________________
#
#         # Define model system
#         system = System(case = "FIX_Fermentation_v2",ode_settings = ode_settings,
#                         loss_settings = loss_settings,rate_settings = rate_settings,dilution = False,
#                         regularization = 1,normalize = True)
#
#         # Create data-set and set up data-shuffler
#         # df = pd.read_excel("Pivot_data_raw_original.xlsx")
#
#         data = Data(case_id = 'FIX_fermentation')
#         included_variables = ["Biomass","Glucose","Glutamine","Lactate","Glutamate","Ammonium",
#                               "FIX_Tank","Osmolality","Offline_pH","Po2"]
#         excluded_variables = ["VCD","Viability","K+","Na+","FIX_harvest","Average_cell_size","Pco2"]
#         units = ["10^6 cells/ml","mM","mM","mM","mM","mM","mg/L","pH units","% of saturation"]
#         stats,mean,stdev,range_var = data.load_from_excel('RAW_DATA_TEST.xlsx',excluded_variables = excluded_variables,
#                                                           included_variables = included_variables)
#         time_series_pair = TimeSeriesPair(data = data,system = system)
#
#         # Split training and validation data
#         training_batches = [32,44,45,48,42,43]
#         validation_batches = [47]
#         test_batches = [50]
#         data.set_batch_pool(pool_batch_id = training_batches,pool_type = 'Training')
#         data.set_batch_pool(pool_batch_id = validation_batches,pool_type = 'Validation')
#         data.set_batch_pool(pool_batch_id = test_batches,pool_type = 'Test')
#         # training_df = df[df.Batch.isin(training_batches)]
#         # training_df["Type"] = "Training"
#         # validation_df = df[df.Batch.isin(validation_batches)]
#         # validation_df["Type"] = "Validation"
#         # test_df = df[df.Batch.isin(test_batches)]
#         # test_df["Type"] = "Test"
#         # 44 as normal, 32 as abnormal
#
#         # Generate shuffled training and evaluation data
#         training_data = time_series_pair.shuffle(pool_type = ['Training'],min_step = 1,max_step = 1)
#         validation_data = time_series_pair.shuffle(pool_type = ['Validation'],min_step = 1,max_step = 1)
#         test_data = time_series_pair.shuffle(pool_type = ['Test'],min_step = 1,max_step = 1)
#         x_train, y_train = hybrid_model.model_data(training_data,included_variables)
#         x_val, y_val = hybrid_model.model_data(validation_data,included_variables)
#         x_test, y_test = hybrid_model.model_data(test_data,included_variables)
#
# #Predictions (27 days)#############################################################
#         first_conditions = [x_test[1][i][0] for i in range(len(x_test[1]))]
#         first_values = [[float(first_conditions[i])] for i in range(len(first_conditions))]
#
#         # Multi time-step predictions
#         multi_step_results = [[] for i in range(len(x_test[1]))]
#         multi_step_results_rates = [[] for i in range(len(hybrid_model.variable_details['y']))]
#         times = np.insert(np.cumsum(x_test[2]),0,0,axis = 0)
#         times_rates = np.cumsum(x_test[2])
#
#
#         for t in range(47):
#
#             measured_constants = [x_test[0][i][t] for i in range(len(x_test[0]))]
#             time = np.array([x_test[2][t]])
#
#             result_rates = hybrid_model.sub_models['rate'].predict(x = [first_conditions])
#             result = hybrid_model.sub_models['ODE'].predict(x = [result_rates,measured_constants,first_conditions,time])
#             print(t,result[6])
#             for i in range(2):
#                 if t==46:
#                     result[i+8] = np.array(x_test[1][i+8][t]).reshape((1,1))
#                 else:
#                     result[i+8] = np.array(x_test[1][i+8][t+1]).reshape((1,1))
#
#             for i in range(len(result_rates)):
#                 multi_step_results_rates[i].append(float(result_rates[i]))
#
#
#
#             first_conditions = result
#             for z in range(len(result)):
#                 multi_step_results[z].append(float(result[z]))
#         #
#         #
#         #
#         # Results plotting
#         for i in range(len(single_step_results)):
#             multi_step_results[i] = first_values[i]+multi_step_results[i]
#
#
#         for i in range(len(multi_step_results_rates)):
#             multi_step_results_rates[i] = np.array(multi_step_results_rates[i])
#
#         multi_step_results_rates = pd.DataFrame(multi_step_results_rates)
#         multi_step_results_rates = pd.DataFrame.transpose(multi_step_results_rates)
#
#         fig,ax = plt.subplots(3,3,sharex = False,sharey = False,figsize = (15,10))
#         for i in range(3):
#             for j in range(3):
#
#
#                 ax[i,j].plot(times,multi_step_results[i * 3+j],label = "Predictions Batch C-0049",lw = 2,
#                              marker = 'o')
#                 ax[i,j].title.set_text(
#                     hybrid_model.variable_details['x'][i * 3+j]['id']+" "+" predicted and real values")
#
#                 ax[i,j].grid()
#                 plt.xticks(range(0,28,2))
#                 ax[i,j].legend(loc = 'best')
#                 plt.gcf().autofmt_xdate()
#
#                 # ax[i,j].title("Model "+"multi step ="+str(a)+" "+"b ="+str(b)+" "+hybrid_model.variable_details['x'][i*3+j]['id'])
#
#         left = 0.125  # the left side of the subplots of the figure
#         right = 0.9  # the right side of the subplots of the figure
#         bottom = 0.1  # the bottom of the subplots of the figure
#         top = 0.9  # the top of the subplots of the figure
#         wspace = 0.2  # the amount of width reserved for blank space between subplots
#         hspace = 0.2  # the amount of height reserved for white space between subplots
#         plt.subplots_adjust(left = left,bottom = bottom,right = right,top = top,wspace = wspace,hspace = hspace)
#         plt.setp(ax[:,:],xlabel = 'Time (Days)')
#         plt.setp(ax[:,:],ylabel = 'Value')
#
#         fig.suptitle("27-Day_Multi-step_prediction "+"a="+str(a)+", b="+str(b))
#
#         plt.savefig("27-Day_Multi-step_prediction.eps",format = "eps")
#         plt.show()
