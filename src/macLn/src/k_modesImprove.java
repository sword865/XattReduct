package macLn.src;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class k_modesImprove {
	
	public static void main(String[] args)throws Exception{

	}

	public static void kModes(Instances data,int k,int[]klable,double[] runPara,String es)throws Exception
	{
		Instance[] centers=new Instance[k];
		for(int i=0;i<k;++i){
			centers[i]=(Instance) data.instance(i).copy();
		}
		FileWriter fileWriter=new FileWriter("E:\\result2\\"+es+"\\iters\\RunTimeParaKi"+".txt",true);
		double minObjectValue=1000000000;
		double F;
		int count=0;
		double[][]weight=new double[k][data.numAttributes()];
//		for(int i=0;i<k;++i){
//			for(int j=0;j<data.numAttributes();++j)
//				weight[i][j]=0.7;
//		}
		k_modes.assign(data,centers,klable);      
		while(true){
			count++;
			//调整中心
			adjustCenters(data,centers,klable,weight);
			//判断是否停止
			F=objectFunction(data,centers,klable,weight);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong1:at kmode_s F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
			assign(data,centers,klable,weight);             //把样本分配到不同中心
			
			//判断是否停止
			F=objectFunction(data,centers,klable,weight);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong2:at kmode_s F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
		}
		fileWriter.append("\r\n");
		fileWriter.flush();
		fileWriter.close();
		runPara[0]=count;
		runPara[1]=k_modes.objectFunction(data,centers,klable);
//		System.out.println("kmodes done");
	}
	
	
	public static double objectFunction(Instances data,Instance[] centers,int[] klable,double[][] weight)
	{
		double gobalDis=0;
		int U=data.numInstances();
		for(int i=0;i<U;++i){
			gobalDis+= dis(i,klable[i],data,centers,weight);
		}
		return gobalDis;
	}
	
	//样本到中心的距离   原始kmode的公式  attr全是categories
	public static double dis(int p,int z,Instances data,Instance[] centers,double[][] weight)
	{
		double distance=0;
	      //如果有类标则不算类标，所以减一 
		for(int i=0;i<data.numAttributes()-1;++i){
			if(data.instance(p).value(i)==centers[z].value(i)){
				distance+= (1-weight[z][i]);
			}else{
				distance+=1;
			}
		}
		return distance;
	}
	
	public static void assign(Instances data,Instance[] centers,int[] klable,double[][] weight)
	{
		int U=data.numInstances();
		int k=centers.length;
		for(int i=0;i<U;++i){
			double mindis=1000000;
			for(int z=0;z<k;++z){
				double d=dis(i,z,data,centers,weight);
				if(d<mindis){
					mindis=d;
					klable[i]=z;
				}
			}
		}
	}
	public static void adjustCenters(Instances data,Instance[] centers,int[] klable,double[][] weight)
	{
		int k=centers.length;
		for(int z=0;z<k;++z){
		    for(int j=0;j<data.numAttributes()-1;++j){
		    	centers[z].setValue(j, maxCate(data,klable,z,j,weight));
			}
		}
	}
	public static double maxCate(Instances data,int[] klable,int z,int j,double[][] weight)
	{
		int U=data.numInstances();
    	int[] cate=new int[50];
    	int num=0;
    	for(int i=0;i<U;++i){
    		if(klable[i]==z){
    			num++;
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
    	if(num==0){
    		weight[z][j]=1.0;
    		System.out.println("wwwki");
    	}else{
    		weight[z][j]=maxnum/num;
    	}
    	return maxcate;
	}
	
}

