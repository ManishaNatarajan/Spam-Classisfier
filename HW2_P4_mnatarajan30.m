% Machine Learning Assignment 2 Q4
% - Manisha Natarajan (GT ID: 903294595)
% Part a) GDA
% Part b) NB GDA
% Part c) NB BDA
% Part d) QDA
%% ==================== Part 1: Initializing ==================
clc
clear all;

%Load data....
data1 = load('spamdata.mat');

%Extracting and Initializing data...
train_class = data1.training_set_label;
train_vect = data1.training_set; 
test_class = data1.testing_set_label;
test_vect = data1.testing_set;
classes = [0;1];%Class labels
num_class = length(classes);
d = size(train_vect,2);%No.of features
n = length(train_class);%No.of samples in training data
nt = length(test_class);%No.of samples in testing data

%% ==================== Part a: GDA ====================
%Part a - Gaussian Distribution........
%Estimating Parameters
mu = zeros(size(train_vect,2),num_class);%[mu0; mu1]
mu(:,1) = ((train_vect)'*(1- train_class))./(n - sum(train_class));
mu(:,2) = ((train_vect)'*(train_class))./sum(train_class);
%Covariance Matrix is a dxd matrix where d = no.of input features
sigma = zeros(size(train_vect,2),size(train_vect,2));
p = sum(train_class == 1)/(length(train_class));
xi_0 = train_vect(train_class == classes(1),:); %Sample corresponding to label 0
xi_1 = train_vect(train_class == classes(2),:); %Sample corresponding to label 1    
for i = 1:length(xi_0)      
   sigma = sigma + (xi_0(i,:)'-mu(:,1))*(xi_0(i,:)'-mu(:,1))';
end
for i = 1:length(xi_1)      
   sigma = sigma + (xi_1(i,:)'-mu(:,2))*(xi_1(i,:)'-mu(:,2))';
end
sigma = sigma./n;

%Prediction....
fprintf('Training Error for GDA');
e = GDA_error(mu,sigma,train_vect,train_class,p);
disp(e);
fprintf('Testing Error for GDA');
e = GDA_error(mu,sigma,test_vect,test_class,p);
disp(e);

%% ==================== Part b: Naive Bayes GDA ===================
%Estimating Parameters
sig_nb = sigma.*eye(d,d);
%mu will be same as GDA
%Prediction....
fprintf('Training Error for NB-GDA');
e = GDA_error(mu,sig_nb,train_vect,train_class,p);
disp(e);
fprintf('Testing Error for NB-GDA');
e = GDA_error(mu,sig_nb,test_vect,test_class,p);
disp(e);
%% ==================== Part c: Naive Bayes BDA ====================
X = zeros(size(train_vect,1),size(train_vect,2));
%Replacing training dataset for Bernoulli Distribution
for i = 1:size(train_vect,1)
    for j = 1:size(train_vect,2)
        if(train_vect(i,j)>0)
            X(i,j) = 1;
        end
    end
end
%Replacing testing dataset for Bernoulli Distribution
for i = 1:size(test_vect,1)
    for j = 1:size(test_vect,2)
        if(test_vect(i,j)>0)
            Xt(i,j) = 1;
        end
    end
end
%Estimating Parameters
p = sum(train_class == 1)/(length(train_class));
for j = 1:size(X,2)
    c=0; a=0;
    for i = 1:n
        if(train_class(i) == 0)
            if(X(i,j) == 1)
                c=c+1;
            end
        end
        if(train_class(i) == 1)
           
            if(X(i,j) == 1)
                a=a+1;
            end
        end
    phi_1(j,:) = a/sum(train_class);
    phi_0(j,:) = c/(n - sum(train_class));     
    end
end
%Prediction
fprintf('Training Error for NB-BDA');
e = BDA_error(phi_0,phi_1,X,train_class,p);
disp(e);
fprintf('Testing Error for NB-BDA');
e = BDA_error(phi_0,phi_1,Xt,test_class,p);
disp(e);

%% ==================== Part d: QDA ====================
%Estimating Parameters
sigma_0 = zeros(size(train_vect,2),size(train_vect,2));
sigma_1 = zeros(size(train_vect,2),size(train_vect,2));
%xi_0 and xi_1 are calculated in GDA
for i = 1:length(xi_0)      
        sigma_0 = sigma_0 + (xi_0(i,:)'-mu(:,1))*(xi_0(i,:)'-mu(:,1))';
     end
     for i = 1:length(xi_1)      
        sigma_1 = sigma_1 + (xi_1(i,:)'-mu(:,2))*(xi_1(i,:)'-mu(:,2))';
     end
 sigma_0 = sigma_0./length(xi_0);
 sigma_1 = sigma_1./length(xi_1);
 %Adding 0.01 to covarince matrix
sigma_0 = sigma_0 + 0.01*eye(d,d);
sigma_1 = sigma_1 + 0.01*eye(d,d);
%Prediction
fprintf('Training Error for QDA');
e = QDA_error(mu,sigma_0,sigma_1,train_vect,train_class,p);
disp(e);
fprintf('Testing Error for QDA');
e = QDA_error(mu,sigma_0,sigma_1,test_vect,test_class,p);
disp(e);

%% ==================== Functions for error calculation ====================
%Please note that for calculating the classification, the expression of $log(p/(1-p))$ where p denotes $P(Y=1|X)$ was used for all of the distributions. 
%If the log value is less than 0, then it is classified as non- spam, else it is spam
function e = GDA_error(m,sig,X,Y,py)
 c = 0; %Counter for correct classifications
 n = size(X,1);
 for j = 1:n
   q(j) =  -0.5*(m(:,1) + m(:,2))' * inv(sig) * (m(:,2) - m(:,1)) + (m(:,2) - m(:,1))'* inv(sig)*X(j,:)' + log(py/(1-py));
   if q(j)<=0
      y_predict(j) = 0;
   else
      y_predict(j) = 1;
   end
end
 for j = 1:n
   if(y_predict(j) == Y(j))
       c = c+1;
   end
end
e = (1 - (c/n))*100;
end

function e = BDA_error(phi0,phi1,X,Y,py)
c = 0;
n = size(X,1);
for j = 1:size(X,1)
    q(j) = sum(log(phi1.*X(j,:)' + (1 - phi1).*(1-X(j,:)')))-sum(log(phi0.*X(j,:)' + (1 - phi0).*(1-X(j,:)'))) + log(py/(1-py));

    if q(j)<0
           y_predict(j) = 0;
       else
           y_predict(j) = 1;
       end
     end
     for j = 1:size(X,1)
         if(y_predict(j) == Y(j))
             c = c+1;
         end
     end
     e = (1 - (c/n))*100;
end
function e = QDA_error(m,sig0,sig1,X,Y,py)
c = 0;
n = size(X,1);
     for j = 1:n
       q(j) =  -0.5*((X(j,:)' - m(:,2))'*inv(sig1)*(X(j,:)' - m(:,2))-(X(j,:)' - m(:,1))'*inv(sig0)*(X(j,:)' - m(:,1)))-0.5*log(det(sig1)/det(sig0))+log(py/(1-py));
       if q(j)<=0
           y_predict(j) = 0;
       else
           y_predict(j) = 1;
       end
     end
     for j = 1:n
         if(y_predict(j) == Y(j))
             c = c+1;
         end
     end
     e = (1 - (c/n))*100;
end