package UFS;

import java.util.Arrays;

import helpLib.Utils_entropy;
import weka.core.Instances;

public class AttributeEvaluation {
	//求symmetrical uncertainty
	//SU(X,Y)
	public static double getSU(Instances dataset, boolean[] X, boolean[] Y){	
		boolean[][] Xi = Utils_entropy.getEquivalenceClass(dataset, X);
		boolean[][] Yj = Utils_entropy.getEquivalenceClass(dataset, Y);
		int U = Xi.length;
		int N = Xi[0].length;
		double Hx = 0.0;
		for(int i=0;i<N;++i){
			int sum=0;
			for(int k=0;k<U;++k){
				if(Xi[k][i])
				sum++;
			}
			if(sum!=(double)0){
				Hx -= sum* (Math.log((double)sum/U) / Math.log((double)2));}
		}
		Hx=(double)Hx/(double)U;
		U = Yj.length;
		N = Yj[0].length;
		double Hy = 0.0;
		for(int i=0;i<N;++i){
			int sum=0;
			for(int k=0;k<U;++k){
				if(Yj[k][i])
				sum++;
			}
			if(sum!=(double)0){
				Hy -= sum* (Math.log((double)sum/U) / Math.log((double)2));}
		}
		Hy=(double)Hy/(double)U;
		U = Xi.length;
		N = Xi[0].length;
		int M = Yj[0].length;
		//计算条件熵
		double res_entropy = 0.0;
		for (int i=0;i<N;++i){				
			for(int j=0; j<M;++j){
				int XandYnum=0;
				int Ynum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[k][j])
						XandYnum++;
					if(Yj[k][j])
						Ynum++;
				}
				if(XandYnum !=0)
					res_entropy=res_entropy-XandYnum* (Math.log((double)XandYnum/Ynum) / Math.log((double)2));
					//注意加(double) 否则part1_sum/part2_sum = 0 熵为负无穷
			}
		}
		
		res_entropy=(double)res_entropy/U;

		double Hxy = res_entropy;
		return 1-2.0*(Hx-Hxy)/(Hx+Hy);
	}
	//双依赖度 pos_A(B)+pos_B(A)/2U
	 static double DoubleDependencyDegree(Instances dataset,boolean[] A, boolean[] B){
		 boolean[][] Ai = Utils_entropy.getEquivalenceClass(dataset, A); //A的等价类
		 boolean[][] Bi = Utils_entropy.getEquivalenceClass(dataset, B); //A的等价类
			
			int N = Ai[0].length;
			int U = Ai.length;
			int M = Bi[0].length;		
			 
			int pos_sum_a = 0;
			for(int i=0;i<N;++i){
				for(int j=0;j<M;++j){
					int cnt = 0;
					for(int k=0;k<U;++k){
						if(Ai[k][i] && Bi[k][j]){
							cnt++; //Xi \in Yj 的个数 [x]_B\in Yj
						}
						if(Ai[k][i] && !Bi[k][j]){//不属于
							cnt=0;
							break;
						}
					}
					pos_sum_a = pos_sum_a + cnt; //pos=\sum|A_| 
				}
			}
			
			int pos_sum_b = 0;
			for(int i=0;i<M;++i){
				for(int j=0;j<N;++j){
					int cnt = 0;
					for(int k=0;k<U;++k){
						if(Bi[k][i] && Ai[k][j]){
							cnt++; //Xi \in Yj 的个数 [x]_B\in Yj
						}
						if(Bi[k][i] && !Ai[k][j]){//不属于
							cnt=0;
							break;
						}
					}
					pos_sum_b = pos_sum_b + cnt;
				}
			}
			
			return 1-(double)(pos_sum_a+pos_sum_b)/(double)(2*U);	
	 }
	 static double DiscriminateDegree(Instances dataset,boolean[] A, boolean[] B){
		 int U = dataset.numInstances();
		//求等价类,先求等价矩阵
		 boolean[][] A_matrix = new boolean[U][U];
		 boolean[][] B_matrix = new boolean[U][U];
		 for(int i=0;i<U;++i){
		Arrays.fill(A_matrix[i], false);
		Arrays.fill(B_matrix[i], false);
		 }
			for(int i=0;i<U;++i){
					for(int j=i+1;j<U;++j){
						for(int k=0;k<A.length;++k){
							if(A[k]&&dataset.instance(i).value(k)!=dataset.instance(j).value(k)){
								A_matrix[i][j]=true;
								A_matrix[j][i]=true;
							}
							if(B[k]&&dataset.instance(i).value(k)!=dataset.instance(j).value(k)){
								B_matrix[i][j]=true;
								B_matrix[j][i]=true;
							}
						}
						
					}
				}
			//A\cap B /   A\cup B
			int cap = 0;
			int cup = 0;
			for(int i=0;i<U;++i){
				cap += AcapB(A_matrix[i],B_matrix[i]);
				cup += AcupB(A_matrix[i],B_matrix[i]);
			}
			if(cup==0)
				return 0;
			else
				return 1-(double)cap/(double)cup;
	}
	 static int AcapB(boolean[] A, boolean[] B)
	 {
		 int cnt = 0;
		 for(int i=0;i<A.length;++i)
		 {
			 if(A[i]&&B[i])
				 cnt++;
		 }
		 return cnt;
	 }
	 static int AcupB(boolean[] A, boolean[] B)
	 {
		 int cnt = 0;
		 for(int i=0;i<A.length;++i)
		 {
			 if(A[i]||B[i])
				 cnt++;
		 }
		 return cnt;
	 }
	 
}
