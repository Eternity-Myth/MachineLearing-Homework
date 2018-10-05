clear all
clc
dataset=[1	0.697 	0.460 	1;
    2 	0.774 	0.376 	1;
    3 	0.634 	0.264 	1;
    4 	0.608 	0.318 	1;
    5 	0.556 	0.215 	1;
    6 	0.403 	0.237 	1;
    7 	0.481 	0.149 	1;
    8 	0.437 	0.211 	1;
    9 	0.666 	0.091 	0;
    10 	0.243 	0.267 	0;
    11 	0.245 	0.057 	0;
    12 	0.343 	0.099 	0;
    13 	0.639 	0.161 	0;
    14 	0.657 	0.198 	0;
    15 	0.360 	0.370 	0;
    16 	0.593 	0.042 	0;
    17 	0.719 	0.103 	0];
X=dataset(:,(2:3));%Xȡ�ܶ��뺬������������
Y=dataset(:,4);%1��ʾ�Ǻùϣ�0��ʾ���Ǻù�

E=0;
for times=1:10
    posrand=randperm(8);
    negrand=randperm(8,8)+8;
    %ȥ�����һ����������4�۽�����֤��һ����Ҫ����4���Ӽ�
    %ȥ�����һ����������������Ϊ8�������ÿ���Ӽ�Ӧ������������2��
    subset1=[dataset(posrand(1,1:2),2:4);dataset(negrand(1,1:2),2:4)];%�Ӽ�1
    subset2=[dataset(posrand(1,3:4),2:4);dataset(negrand(1,3:4),2:4)];%�Ӽ�2
    subset3=[dataset(posrand(1,5:6),2:4);dataset(negrand(1,5:6),2:4)];%�Ӽ�3
    subset4=[dataset(posrand(1,7:8),2:4);dataset(negrand(1,7:8),2:4)];%�Ӽ�4

    %�����������ֲ�ͼ
    figure;
    hold on;
    pos=find(Y==1);%����
    neg=find(Y==0);%����
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%�������㣬�á�+����ʾ
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%�������㣬�á�������ʾ
    xlabel('�ܶ�')
    ylabel('������')

    training1=[subset1;subset2;subset3];%��һ�Σ�ȡǰ3���Ӽ���ѵ��������4���Ӽ������Լ�
    X1=training1(:,1:2);%Xȡ�ܶ��뺬������������
    Y1=training1(:,3);%1��ʾ�Ǻùϣ�0��ʾ���Ǻù�
    [m,n]=size(X1);
    X1=[X1,ones(m,1)];%�����Ҷ����1��1����ϳ�����b
    initial_theta=zeros(n+1,1);%��ʼ��ϵ��theta
    options=optimset('GradObj','on','MaxIter',400);
    %����fminunc��������߼��ع����Ѳ�����Ҳ����ʹcostFunction�ﵽ��Сֵ�Ķ�Ӧ����theta
    [theta1,cost1]=fminunc(@(t)(costFunction(t, X1, Y1)),initial_theta,options);
    %��4���Ӽ����ڲ���
    X4=subset4(:,1:2);
    X4=[X4,ones(4,1)];
    %������ȷ��
    count=0;
    for i=1:4
        if((X4(i,:)*theta1>0&&subset4(i,3)==1)||(X4(i,:)*theta1<0&&subset4(i,3)==0))
            count=count+1;
        end
    end
    e1=count/4;

    %�����ж��߽�ֱ��
    x=0:0.1:0.8;
    line=(-theta1(3,1)-theta1(1,1)*x)/theta1(2,1);%�ж��߽�ֱ�߷���Ϊ��theta1*x1+theta2*x2+theta3=0�����Է����theta2��б��ʽ����
    plot(x,line);
    str=strcat('y=',num2str(-theta1(1,1)/theta1(2,1)),'x+',num2str(-theta1(3,1)/theta1(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %ʹ��text������ͼ�ϱ�ע��ֱ�߷���


    %�����������ֲ�ͼ
    figure;
    hold on;
    pos=find(Y==1);%����
    neg=find(Y==0);%����
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%�������㣬�á�+����ʾ
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%�������㣬�á�������ʾ
    xlabel('�ܶ�')
    ylabel('������')

    training2=[subset1;subset2;subset4];%�ڶ��Σ�ȡ1 2 4�Ӽ���ѵ��������3���Ӽ������Լ�
    X2=training2(:,1:2);%Xȡ�ܶ��뺬������������
    Y2=training2(:,3);%1��ʾ�Ǻùϣ�0��ʾ���Ǻù�
    [m,n]=size(X2);
    X2=[X2,ones(m,1)];%�����Ҷ����1��1����ϳ�����b
    initial_theta=zeros(n+1,1);%��ʼ��ϵ��theta
    options=optimset('GradObj','on','MaxIter',400);
    %����fminunc��������߼��ع����Ѳ�����Ҳ����ʹcostFunction�ﵽ��Сֵ�Ķ�Ӧ����theta
    [theta2,cost2]=fminunc(@(t)(costFunction(t, X2, Y2)),initial_theta,options);
    %��3���Ӽ����ڲ���
    X3=subset3(:,1:2);
    X3=[X3,ones(4,1)];
    %������ȷ��
    count=0;
    for i=1:4
        if((X3(i,:)*theta2>0&&subset3(i,3)==1)||(X3(i,:)*theta2<0&&subset3(i,3)==0))
            count=count+1;
        end
    end
    e2=count/4;

    %�����ж��߽�ֱ��
    x=0:0.1:0.8;
    line=(-theta2(3,1)-theta2(1,1)*x)/theta2(2,1);%�ж��߽�ֱ�߷���Ϊ��theta1*x1+theta2*x2+theta3=0�����Է����theta2��б��ʽ����
    plot(x,line);
    str=strcat('y=',num2str(-theta2(1,1)/theta2(2,1)),'x+',num2str(-theta2(3,1)/theta2(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %ʹ��text������ͼ�ϱ�ע��ֱ�߷���

    %�����������ֲ�ͼ
    figure;
    hold on;
    pos=find(Y==1);%����
    neg=find(Y==0);%����
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%�������㣬�á�+����ʾ
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%�������㣬�á�������ʾ
    xlabel('�ܶ�')
    ylabel('������')

    training3=[subset1;subset3;subset4];%�����Σ�ȡ1 3 4��ѵ��������2���Ӽ������Լ�
    X3=training3(:,1:2);%Xȡ�ܶ��뺬������������
    Y3=training3(:,3);%1��ʾ�Ǻùϣ�0��ʾ���Ǻù�
    [m,n]=size(X3);
    X3=[X3,ones(m,1)];%�����Ҷ����1��1����ϳ�����b
    initial_theta=zeros(n+1,1);%��ʼ��ϵ��theta
    options=optimset('GradObj','on','MaxIter',400);
    %����fminunc��������߼��ع����Ѳ�����Ҳ����ʹcostFunction�ﵽ��Сֵ�Ķ�Ӧ����theta
    [theta3,cost3]=fminunc(@(t)(costFunction(t, X3, Y3)),initial_theta,options);
    %��2���Ӽ����ڲ���
    X2=subset2(:,1:2);
    X2=[X2,ones(4,1)];
    %������ȷ��
    count=0;
    for i=1:4
        if((X2(i,:)*theta3>0&&subset2(i,3)==1)||(X2(i,:)*theta3<0&&subset2(i,3)==0))
            count=count+1;
        end
    end
    e3=count/4;

    %�����ж��߽�ֱ��
    x=0:0.1:0.8;
    line=(-theta3(3,1)-theta3(1,1)*x)/theta3(2,1);%�ж��߽�ֱ�߷���Ϊ��theta1*x1+theta2*x2+theta3=0�����Է����theta2��б��ʽ����
    plot(x,line);
    str=strcat('y=',num2str(-theta3(1,1)/theta3(2,1)),'x+',num2str(-theta3(3,1)/theta3(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %ʹ��text������ͼ�ϱ�ע��ֱ�߷���

    %�����������ֲ�ͼ
    figure;
    hold on;
    pos=find(Y==1);%����
    neg=find(Y==0);%����
    plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);%�������㣬�á�+����ʾ
    plot(X(neg,1),X(neg,2),'o','MarkerFaceColor','r','MarkerSize',7);%�������㣬�á�������ʾ
    xlabel('�ܶ�')
    ylabel('������')

    training4=[subset2;subset3;subset4];%���ĴΣ�ȡ2 3 4��ѵ��������1���Ӽ������Լ�
    X4=training4(:,1:2);%Xȡ�ܶ��뺬������������
    Y4=training4(:,3);%1��ʾ�Ǻùϣ�0��ʾ���Ǻù�
    [m,n]=size(X4);
    X4=[X4,ones(m,1)];%�����Ҷ����1��1����ϳ�����b
    initial_theta=zeros(n+1,1);%��ʼ��ϵ��theta
    options=optimset('GradObj','on','MaxIter',400);
    %����fminunc��������߼��ع����Ѳ�����Ҳ����ʹcostFunction�ﵽ��Сֵ�Ķ�Ӧ����theta
    [theta4,cost4]=fminunc(@(t)(costFunction(t, X4, Y4)),initial_theta,options);
    %��1���Ӽ����ڲ���
    X1=subset1(:,1:2);
    X1=[X1,ones(4,1)];
    %������ȷ��
    count=0;
    for i=1:4
        if((X1(i,:)*theta4>0&&subset1(i,3)==1)||(X1(i,:)*theta4<0&&subset1(i,3)==0))
            count=count+1;
        end
    end
    e4=count/4;

    %�����ж��߽�ֱ��
    x=0:0.1:0.8;
    line=(-theta4(3,1)-theta4(1,1)*x)/theta4(2,1);%�ж��߽�ֱ�߷���Ϊ��theta1*x1+theta2*x2+theta3=0�����Է����theta2��б��ʽ����
    plot(x,line);
    str=strcat('y=',num2str(-theta4(1,1)/theta4(2,1)),'x+',num2str(-theta4(3,1)/theta4(2,1)));
    text(0.1,0.45,str,'interpreter','latex','fontsize',20) %ʹ��text������ͼ�ϱ�ע��ֱ�߷���

    averageE=(e1+e2+e3+e4)/4;
    E=E+averageE;
end
E/10

function [J, grad] = costFunction(theta, X, Y)
m = length(Y); %ѵ����������
grad = zeros(size(theta));%��ʼ���ݶ�Ϊ0
h=1.0./(1.0+exp(-1*X*theta));%�������ʺ�������Sigmoid����
m=size(Y,1);
J=((-1*Y)'*log(h)-(1-Y)'*log(1-h))/m;%�򻯺�ͳһ��ʽ��costFunction J
for i=1:size(theta,1),
    grad(i)=((h-Y)'*X(:,i))/m;%��costFunction J�󵼿ɵ��ݶȵı��ʽ
end
end