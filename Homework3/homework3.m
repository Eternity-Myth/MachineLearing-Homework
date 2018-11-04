clear
clear all
%��ÿ�����Խ��и�ֵ��
%ɫ������-1 �ں�-2 ǳ��-3
%���٣�����-1 ����-2 Ӳͦ-3
%����������-1 ����-2 ���-3
%��������-1 �Ժ�-2 ģ��-3
%�겿������-1 �԰�-2 ƽ̹-3
%���У�Ӳ��-1 ��ճ-2
%�ùϣ���-1 ��-0
X=[1	1	1	1	1	1;
2	1	2	1	1	1;
2	1	1	1	1	1;
1	2	1	1	2	2;
2	2	1	2	2	2;
1	3	3	1	3	2;
3	2	2	2	1	1;
2	2	1	1	2	2;
3	1	1	3	3	1;
1	1	2	2	2	1;
1	1	2	1	1	1;
3	1	1	1	1	1;
2	2	1	1	2	1;
2	2	2	2	2	1;
3	3	3	3	3	1;
3	1	1	3	3	2;
1	2	1	2	1	1];
Y=[1;1;1;1;1;0;0;0;0;0;1;1;1;0;0;0;0];
trainingX=X(1:10,:);%ȡǰ10������Ϊѵ����
trainingY=Y(1:10,:);%ȡǰ10������Ϊѵ����
testX=X(11:17,:);%ȡ��7������Ϊ���Լ�
testY=Y(11:17,:);%ȡ��7������Ϊ���Լ�

[row,column]=size(trainingX);%row�Ǿ������������ʾ�ܹ��ж��ٸ�ѵ������column�Ǿ������������ʾѵ���������롣
OutputLayerNum=1;%�������Ԫ��
v=rand(column,column+1);%������������Ȩֵ,v��һ��column��column+1�о���
w=rand(column+1,OutputLayerNum);%������������Ȩֵ,w��һ��column+1��1�о���
gamma=rand(column+1);%������ֵ,gamma��column+1��1�о���
theta=rand(OutputLayerNum);%�������ֵ,theta��1��1�о���
output=zeros(row,OutputLayerNum);%��������
b=zeros(column+1);%�������
g=zeros(OutputLayerNum);%��������w,gamma�󵼵Ĳ���
e=zeros(column+1);%��������v,theta�󵼵Ĳ���
LearningRate=0.1;%ѧϰ�ʣ���0-1֮��
IterativeTimes=0;%�����Ĵ���
AccumulateTimes=0;%ͬ���ľ������ֵ�ۻ�����
previous_E=0;%ǰһ�ε������ۼ����

while(1)
    IterativeTimes=IterativeTimes+1;
    E=0;%��ǰ�����ľ������
    %����ȫ��������������
    for i=1:row
      %������������
      for j=1:column+1
        alpha=0;
        for k=1:column
          alpha=alpha+v(k,j)*trainingX(i,k);
        end
        b(i,j)=1/(1+exp(-alpha+gamma(j)));%����sigmoid����
      end
       %������������
       for j=1:OutputLayerNum
         beta=0;
         for k=1:column+1
           beta=beta+w(k,j)*b(i,k);
         end
          output(i,j)=1/(1+exp(-beta+theta(j)));%����sigmoid����
       end
    end
    %�����洢�ۻ������ĸ��������½����򣬼�delta��
    delta_v=zeros(column,column+1);
    delta_w=zeros(column+1,OutputLayerNum);
    delta_gamma=zeros(column+1);
    delta_theta=zeros(OutputLayerNum);
    %�����ۻ����
    for i=1:row
        for j=1:OutputLayerNum
          E=E+((trainingY(i)-output(i,j))^2)/2;%�������E
        end
        %����w��theta��������
        for j=1:OutputLayerNum
          g(j)=output(i,j)*(1-output(i,j))*(trainingY(i)-output(i,j));%����
        end
        %����v��gamma��������
        for j=1:column+1
          teh=0;
          for k=1:OutputLayerNum
            teh=teh+w(j,k)*g(k);
          end
            e(j)=teh*b(i,j)*(1-b(i,j));%����
        end
        %����w��theta����
        for j=1:OutputLayerNum
          delta_theta=delta_theta+(-1)*LearningRate*g(j);
          for k=1:column+1
            delta_w(k,j)=delta_w(k,j)+LearningRate*g(j)*b(i,k);
          end
        end
        %����v��gamma����
        for j=1:column+1
          gamma(j)= gamma(j)+(-1)*LearningRate*e(j);
          for k=1:column
            delta_v(k,j)=delta_v(k,j)+LearningRate*e(j)*trainingX(i,k);
          end
        end
    end
    %���²���
    v=v+delta_v;
    w=w+delta_w;
    gamma=gamma+delta_gamma;
    theta=theta+delta_theta;
    %���õ�����ֹ������ǰ���������֮�����ֵС��0.01%�����ۼ�500��
    if(abs(previous_E-E)<0.0001)
      AccumulateTimes=AccumulateTimes+1;
      if(AccumulateTimes==500)%���λ���趨��Χ���ۼ�500��
        break;
      end
    else
      previous_E=E;
      AccumulateTimes=0;
   end
end
testoutput=zeros(7,OutputLayerNum);%���Լ���������
testb=zeros(column+1);%���Լ��������
for i=1:7
      %������Լ���������
      for j=1:column+1
        alpha=0;
        for k=1:column
          alpha=alpha+v(k,j)*testX(i,k);
        end
        testb(i,j)=1/(1+exp(-alpha+gamma(j)));%����sigmoid����
      end
       %������Լ���������
       for j=1:OutputLayerNum
         beta=0;
         for k=1:column+1
           beta=beta+w(k,j)*testb(i,k);
         end
          testoutput(i,j)=1/(1+exp(-beta+theta(j)));%����sigmoid����
       end
end
%������Լ��������
testE=0;
for i=1:7
  for j=1:OutputLayerNum
    testE=testE+((testY(i)-testoutput(i,j))^2)/2;
  end
end