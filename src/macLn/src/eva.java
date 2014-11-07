package macLn.src;

import weka.core.Instances;


public class eva {
	public static double AC(int U,int k,int[]a)
	{
		double sum=0;
		for(int z=0;z<k;++z){
			sum+=a[z];
		}
		return sum/U;
	}
	
	public static double PE(int k,int[]a,int[]b)
	{
		double sum=0;
		for(int z=0;z<k;++z){
			if(a[z]==0) return 0.0;
			sum+=((double)a[z]/(a[z]+b[z]));
		}
		return sum/k;
	}
	public static double RE(int k,int[]a,int[]c)
	{
		return PE(k,a,c);
	}
	public static double Sum(double v[])
	{	
		double s=0;
		for(int i=0;i<v.length;++i){
			s+=v[i];
		}
		return s;
	}
	public static int Sum(int[] v) {
		// TODO Auto-generated method stub
		int s=0;
		for(int i=0;i<v.length;++i){
			s+=v[i];
		}
		return s;
	}
	public static double Mean(double v[])
	{
		return Sum(v)/v.length;
	}
	public static double standardDev(double v[])
	{
		double mean=Mean(v);
		double rs=0;
		for(int i=0;i<v.length;++i){
			rs+=(v[i]-mean)*(v[i]-mean);
		}
		rs/=v.length;
		return rs;
	}
	
	
	public static double NMI(int[]X,int[]Y,int k)
	{
		double tmp=entropy(X,k)+entropy(Y,k);
		return 2*(tmp-jointEntropy(X,Y,k))/(tmp);
	}
	
	public static double entropy(int[]X,int k)
	{
		int[] nums=new int[k];
		double sum=0;
		for(int i=0;i<X.length;++i){
			nums[X[i]]++;
		}
		for(int i=0;i<k;++i){
			if(nums[i]!=0){
				double t=(double)nums[i]/X.length;
				sum+= t*Math.log(t);
			}
		}
		return -sum/Math.log(2);
	}
//	public static double mutualInfo(int[]X,int[]Y)
//	{
//		
//	}
	
	public static double jointEntropy(int[]X,int[]Y,int k)
	{
		int[][] mat=new int[k][k];
		for(int i=0;i<X.length;++i){
				mat[X[i]][Y[i]]++;
		}
		double sum=0;
		for(int i=0;i<k;++i){
			for(int j=0;j<k;++j){
				if(mat[i][j]!=0){
					double t=(double)mat[i][j]/X.length;
					sum+= t*Math.log(t);
				}
			}
		}
		return -sum/Math.log(2);
	}
	public static double ARI(int[]C,int[]T)
	{
		//C算法得到的分类， T真实分类
		long a=0;  //pairs in C in T 
		long b=0;  //pairs not in C in T
		long c=0;  //pairs in C not in T
		long d=0;  //pairs not in C not in T
		int U=C.length;
		for(int i=0;i<U-1;++i){
			for(int j=i+1;j<U;++j){
				if((C[i]==C[j]) && (T[i]==T[j]) ){
					a++;
				}else if((C[i]!=C[j]) && (T[i]==T[j])){
					b++;
				}else if((C[i]==C[j]) && (T[i]!=T[j])){
					c++;
				}else{
					d++;
				}
			}
		}
		return (2.0*(a*d-b*c))/((a+b)*(b+d)+(a+c)*(c+d));
	}
	
	public static double MS(int[]C,int[]T)
	{
		//C算法得到的分类， T真实分类
		long a=0;  //pairs in C in T 
		long b=0;  //pairs not in C in T
		long c=0;  //pairs in C not in T
		long d=0;  //pairs not in C not in T
		long U=C.length;
		for(int i=0;i<U-1;++i){
			for(int j=i+1;j<U;++j){
				if((C[i]==C[j]) && (T[i]==T[j]) ){
					a++;
				}else if((C[i]!=C[j]) && (T[i]==T[j])){
					b++;
				}else if((C[i]==C[j]) && (T[i]!=T[j])){
					c++;
				}else{
					d++;
				}
			}
		}
		return (double)(2.0*(b+c))/(2*(a+b));
	}

	public static void clusterAnalysis(Instances data,int[]klable,int[]rslable,int k,int[]a,int[]b,int[]c)
	{
		int U=data.numInstances();
		int p=data.classIndex();
		int[] countLable=new int[data.numClasses()];
		for(int j=0;j<U;++j){
			countLable[(int)data.instance(j).value(p)]++;
		}
		for(int j=0;j<U;++j){
			c[klable[j]]=countLable[rslable[klable[j]]];
		}
		for(int i=0;i<U;++i){
			if(data.instance(i).value(p)==rslable[klable[i]]){
				a[klable[i]]++;
			}else{
				b[klable[i]]++;
			}
		}
		for(int i=0;i<k;++i){
			c[i]-=a[i];
		}
	}
	public static double[] evalution(Instances data,int[] klable,int k,double[] rs)
	{
		//rs数组从0开始依次对应ac,pe,re,nmi,ari,ms的指标
		int U=data.numInstances();
		int[] a=new int[k];
		int[] b=new int[k];
		int[] c=new int[k];
		int[] tlable=new int[U];
		int[] rslable=new int[U];
		utility.maketLable(data,tlable,U);
		utility.assignLable(data,klable,rslable,k);
		clusterAnalysis(data,klable,rslable,k,a,b,c);
		rs[0]=AC(U,k,a);
		rs[1]=PE(k,a,b);
		rs[2]=RE(k,a,c);
		rs[3]=NMI(klable,tlable,data.numClasses());
		rs[4]=ARI(klable,tlable);
		rs[5]=MS(klable,tlable);
		return rs;
	}

	
}
