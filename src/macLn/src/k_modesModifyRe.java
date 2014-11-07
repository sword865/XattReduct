package macLn.src;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class k_modesModifyRe {
	public static void main(String[] args)throws Exception
	{
	 
	}
	public static void kModes(Instances data,int k,int[] klable,double[] runPara,String es) throws IOException
	{
		Instance[] centers=new Instance[k];
		Instance[] centers2=new Instance[k];
		for(int i=0;i<k;++i){
			centers[i]=(Instance) data.instance(i).copy();
			centers2[i]=(Instance)data.instance(i).copy();
		}
		
		FileWriter fileWriter=new FileWriter("E:\\result2\\"+es+"\\iters\\RunTimeParaKmr"+".txt",true);
		double minObjectValue=1000000000;
		double F;
		int count=0;
		int[][][] countAttr=new int[k][data.numAttributes()][50];
		k_modes.assign(data,centers,klable);      
		while(true){
			count++;
			//调整中心
			adjustCentersNew(data,centers,centers2,countAttr,klable);
		//判断是否停止
			F=objectFunction(data,centers,centers2,countAttr,klable);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong1:at kmodesModify F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
			
			assignNew(data,centers,centers2,countAttr,klable);             //把样本分配到不同中心
			//判断是否停止
			F=objectFunction(data,centers,centers2,countAttr,klable);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong2:at kmodesModify F:"+F+"  "+minObjectValue );
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
//		System.out.println("kmodesModify done");
	}
	
	
	public static double objectFunction(Instances data,Instance[] centers,Instance[] centers2,int[][][]countAttr,int[] klable)
	{
		double gobalDis=0;
		int U=data.numInstances();
		for(int i=0;i<U;++i){
			gobalDis+= disNew(i,klable[i],data,countAttr,centers,centers2);
		}
		return gobalDis;
	}
	
	//样本到中心的距离   原始kmode的公式  attr全是categories
	public static double disNew(int p,int z,Instances data,int[][][]countAttr,Instance[] centers,Instance[] centers2)
	{
		double distance=0;
	      //如果有类标则不算类标，所以减一 
		double s=eva.Sum(countAttr[z][0]);
		if(s==0.0){
			s=1024;
			System.out.println("wwwmr");
		}
		for(int i=0;i<data.numAttributes()-1;++i){
			double v=data.instance(p).value(i);
			if((v!=centers[z].value(i))&& 
					(v!=centers2[z].value(i)) ){
				distance+=1-countAttr[z][i][(int)v]/s;
			}
		}
		return distance;
	}
	public static void assignNew(Instances data,Instance[] centers,Instance[] centers2,int[][][]countAttr,int[] klable)
	{
		int U=data.numInstances();
		int k=centers.length;
		for(int i=0;i<U;++i){
			double mindis=5000000;
			for(int z=0;z<k;++z){
				double d=disNew(i,z,data,countAttr,centers,centers2);
				if(d<mindis){
					mindis=d;
					klable[i]=z;
				}
			}
		}
	}
	public static void adjustCentersNew(Instances data,Instance[] centers,Instance[] centers2,int[][][]countAttr,int[] klable)
	{
		int k=centers.length;
		for(int z=0;z<k;++z){
		    for(int j=0;j<data.numAttributes()-1;++j){
		    	double[] d=maxCate(data,klable,countAttr,z,j);
		    	centers[z].setValue(j, d[0]);
		    	centers2[z].setValue(j, d[1]);
			}
		}
	}
	public static double[] maxCate(Instances data,int[] klable,int[][][]countAttr,int z,int j)
	{
		int U=data.numInstances();
    	Arrays.fill(countAttr[z][j], 0);
    	for(int i=0;i<U;++i){
    		if(klable[i]==z){
    			double s=data.instance(i).value(j);
    			if(s>50){
    				System.out.print(data.attribute(j));
    			}
    			countAttr[z][j][(int)(s)]++;
			}	
		}
    	
    	double maxnum1=0,num=0,maxnum2=0;
    	double[] maxcate=new double[3];
    	maxcate[0]=maxcate[1]=-1;
    	for(int i=0;i<50;++i){
    		if(countAttr[z][j][i]>0){
    			num++;
    		}
    		if(countAttr[z][j][i]>maxnum1){
    			maxcate[0]=i;
    			maxnum1=countAttr[z][j][i];
    		}
    	}
    	if(num<4){//data.attribute(j).numValues()<4){
    		maxcate[1]=maxcate[0];
    		//maxnum2=maxnum1;
    	}else{
	    	for(int i=0;i<50;++i){
	    		if(countAttr[z][j][i]>maxnum2 && i!=maxcate[0]){
	    			maxcate[1]=i;
	    			maxnum2=countAttr[z][j][i];
	    		}
	    	}
    	}
		//maxcate[2]= maxnum1/(maxnum1+maxnum2);
    	return maxcate;
	}
	
}

