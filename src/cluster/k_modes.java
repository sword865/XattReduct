package cluster;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class k_modes {
	
	public static void main(String[] args)throws Exception{

	}

	public static double kModes(Instances data,int k,int[]klable,double[] runPara,Random rand)throws Exception
	{

		Instance[] centers=new Instance[k];
		for(int i=0;i<k;++i){
			centers[i]=(Instance) data.instance(i).copy();
		}
//		FileWriter fileWriter=new FileWriter("E:\\result2\\"+es+"\\iters\\RunTimePara"+".txt",true);
		double minObjectValue=1000000000;
		double F;
		int count=0;
		assign(data,centers,klable);             //把样本分配到不同中心
		while(true){
			count++;
			//调整中心
			adjustCenters(data,centers,klable);
			//判断是否停止
			F=objectFunction(data,centers,klable);
//			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong1:at kmode_s F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
			assign(data,centers,klable);             //把样本分配到不同中心
			
			//判断是否停止
			F=objectFunction(data,centers,klable);
//			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong2:at kmode_s F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
		}
//		fileWriter.append("\r\n");
//		fileWriter.flush();
//		fileWriter.close();
		runPara[0]=count;
		runPara[1]=k_modes.objectFunction(data,centers,klable);
		return runPara[1];
//		System.out.println("kmodes done");
	}
	
	
	public static double objectFunction(Instances data,Instance[] centers,int[] klable)
	{
		double gobalDis=0;
		int U=data.numInstances();
		for(int i=0;i<U;++i){
			gobalDis+= dis(i,klable[i],data,centers);
		}
		return gobalDis;
	}
	
	//样本到中心的距离   原始kmode的公式  attr全是categories
	public static double dis(int p,int z,Instances data,Instance[] centers)
	{
		double distance=0;
	      //如果有类标则不算类标，所以减一 
		for(int i=0;i<data.numAttributes()-1;++i){
			if(data.instance(p).value(i)!=centers[z].value(i)){
				distance+=1;
			}
		}
		return distance;
	}
	
	public static void assign(Instances data,Instance[] centers,int[] klable)
	{
		int U=data.numInstances();
		int k=centers.length;
		for(int i=0;i<U;++i){
			double mindis=1000000;
			for(int z=0;z<k;++z){
				double d=dis(i,z,data,centers);
				if(d<mindis){
					mindis=d;
					klable[i]=z;
				}
			}
		}
	}
	public static void adjustCenters(Instances data,Instance[] centers,int[] klable)
	{
		int k=centers.length;
		for(int z=0;z<k;++z){
		    for(int j=0;j<data.numAttributes()-1;++j){
		    	centers[z].setValue(j, maxCate(data,klable,z,j));
			}
		}
	}
	public static double maxCate(Instances data,int[] klable,int z,int j)
	{
		int U=data.numInstances();
    	int[] cate=new int[50];
    	for(int i=0;i<U;++i){
    		if(klable[i]==z){
    			double s=data.instance(i).value(j);
    			cate[(int)(s)]++;
			}	
		}
    	double maxnum=0,maxcate=-1;
    	for(int i=0;i<50;++i){
    		if(cate[i]>maxnum){
    			maxnum=cate[i];
    			maxcate=i;
    		}
    	}
    	return maxcate;
	}
	
//	public static void assignLable(Instances data,int[] klable,int[] rslable,int k)
//	{
//		for(int z=0;z<k;++z){
//			rslable[z]=(int)maxCate(data,klable,z,data.numAttributes()-1);
//		}
//	}
	
}

