########################################################
#libraries
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import tenseal as ts
import phe as paillier

########################################################
# loading model
def model_scaler_heart(m,n):
    with open(m,'rb') as f:
        model_heart=pickle.load(f)
    with open(n, 'rb') as f:
        scaler_heart=pickle.load(f)
    return model_heart,scaler_heart

########################################################
# FHE
def predict_heart_fhe(m,n,input):
    # key generation
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    # model
    model,scaler=model_scaler_heart(m,n)

    # input
    numpy_input = np.asarray(input)
    input_reshaped = numpy_input.reshape(1,-1)
    std_data = scaler.transform(input_reshaped)
    inp=std_data.flatten()
    

    # encryption
    enc_input = ts.ckks_vector(context, inp)

    # coeff,intercept
    coef = model.coef_.flatten()
    coef = coef.reshape(-1)
    enc_coef= ts.ckks_vector(context, coef)

    enc_in=ts.ckks_vector(context, model.intercept_)

    # prediction
    prediction=enc_input.dot(enc_coef)
    prediction += enc_in

    # decryption

    if(prediction.decrypt()[0] >= 0):
        return 'The person has heart  disease'
    else:
        return 'The person does not have heart disease'


########################################################
# PHE
def predict_heart_phe(m,n,input):
    # key generation
    public_key, private_key = paillier.generate_paillier_keypair()

    # model
    model,scaler=model_scaler_heart(m,n)

    numpy_input = np.asarray(input)
    input_reshaped = numpy_input.reshape(1,-1)
    std_data = scaler.transform(input_reshaped)
    inp=std_data.flatten()
    input=inp.tolist()

    # encryption
    enc_input=[]
    for i in input:
        enc_input.append(public_key.encrypt(i))

    numpy_input = np.asarray(enc_input)
    
    # coeff,intercept
    coef = model.coef_.flatten()
    coef = coef.reshape(-1)

    inter = model.intercept_[0]

    # prediction
    prediction=0
    for i in range(0,len(coef)):
        prediction+=(enc_input[i] * coef[i])
    prediction+= inter

    # decryption

    if(private_key.decrypt(prediction) >= 0):
        return 'The person has heart  disease'
    else:
        return 'The person does not have heart disease'


########################################################
# DP 

def add_laplace_noise(data, epsilon, reduction_factor):
    sensitivity = 1 
    original_scale = sensitivity / epsilon
    reduced_scale = original_scale * reduction_factor
    
    noisy_data = []
    for element in data:
        noisy_data.append(element + np.random.laplace(scale=reduced_scale))

    return noisy_data

def predict_heart_dp(m,n,input):

    # model
    model,scaler=model_scaler_heart(m,n)

    # input
    numpy_input = np.asarray(input)
    input_reshaped = numpy_input.reshape(1,-1)
    std_data = scaler.transform(input_reshaped)
    inp=std_data.flatten()

    # Add noise in the input
    noisy_input = add_laplace_noise(inp, 1, 0.5)
    noisy_input = np.array(noisy_input)

    # coeff,intercept
    coef = model.coef_.flatten()
    coef = coef.reshape(-1)

    inter = model.intercept_[0]

    # prediction
    prediction=0
    for i in range(0,len(coef)):
        prediction+=(noisy_input[i] * coef[i])
    prediction+= inter

    # decryption

    if(prediction >= 0):
        return 'The person has heart  disease'
    else:
        return 'The person does not have heart disease'

########################################################
# FHE with DP 

def predict_heart_fhe_dp(m,n,input):

    # key generation
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree = 8192, coeff_mod_bit_sizes = [60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40

    # model
    model,scaler=model_scaler_heart(m,n)

    # input
    numpy_input = np.asarray(input)
    input_reshaped = numpy_input.reshape(1,-1)
    std_data = scaler.transform(input_reshaped)
    inp=std_data.flatten()

    # Add noise in the input
    noisy_input = add_laplace_noise(inp, 1, 0.5)
    noisy_input = np.array(noisy_input)

    # encryption
    enc_input = ts.ckks_vector(context, noisy_input)

    # coeff,intercept
    coef = model.coef_.flatten()
    coef = coef.reshape(-1)
    enc_coef= ts.ckks_vector(context, coef)

    enc_in=ts.ckks_vector(context, model.intercept_)

    # prediction
    prediction=enc_input.dot(enc_coef)
    prediction += enc_in

    # decryption

    if(prediction.decrypt()[0] >= 0):
        return 'The person has heart disease'
    else:
        return 'The person does not have heart disease'

########################################################
# prediction without any encryption
def predict_heart(m,n,input):
    model,scaler=model_scaler_heart(m,n)

    numpy_input = np.asarray(input)

    # reshape array for predicting 1 instance
    input_reshaped = numpy_input.reshape(1,-1)

    std_data = scaler.transform(input_reshaped)
    prediction = model.predict(std_data)

    if(prediction[0] == 0):
        return 'The person does not have heart  disease'
    else:
        return 'The person has heart disease'
