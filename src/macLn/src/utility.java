package macLn.src;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import weka.core.Instances;


public class utility {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	public static double[] maxCate(Instances data,int[] klable,int z,int j)
	{
		int U=data.numInstances();
    	int[] cate=new int[50];
    	for(int i=0;i<U;++i){
    		if(klable[i]==z){
    			double s=data.instance(i).value(j);
    			if(s>50){
    				System.out.print(data.attribute(j));
    			}
    			cate[(int)(s)]++;
			}	
		}
    	
    	double maxnum=0,num=0;
    	double[] maxcate=new double[2];
    	maxcate[0]=maxcate[1]=-1;
    	for(int i=0;i<50;++i){
    		if(cate[i]>0){
    			num++;
    		}
    		if(cate[i]>maxnum){
    			maxcate[0]=i;
    			maxnum=cate[i];
    		}
    	}
    	if(num<4){//data.attribute(j).numValues()<4){
    		maxcate[1]=maxcate[0];
    	}else{
    		maxnum=-1;
	    	for(int i=0;i<50;++i){
	    		if(cate[i]>maxnum && i!=maxcate[0]){
	    			maxcate[1]=i;
	    			maxnum=cate[i];
	    		}
	    	}
    	}
    	return maxcate;
	}
	public static void assignLable(Instances data,int[] klable,int[] rslable,int k)
	{
		for(int z=0;z<k;++z){
			rslable[z]=(int)maxCate(data,klable,z,data.classIndex())[0];
		}
	}
	public static void maketLable(Instances data,int[]tlable,int U)
	{
		int p=data.classIndex();
		for(int i=0;i<U;++i){
			tlable[i]=(int)data.instance(i).value(p);
		}
	}
	
	public static void setVec(int[] v,int a)
	{
		for(int i=0;i<v.length;++i){
			v[i]=a;
		}
	}
	public static void sortValues(double[]v,int[]r)
	{
		for(int i=0;i<r.length;++i){
			r[i]=i;
		}
		for(int i=0;i<v.length;++i){
			for(int j=i+1;j<v.length;++j){
				if(v[i]>v[j]){
					double dt;
					dt=v[i];
					v[i]=v[j];
					v[j]=dt;
					int it;
					it=r[i];
					r[i]=r[j];
					r[j]=it;
				}
			}
		}
	}
	
	
	
	//public static void 
}
