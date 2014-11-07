package helpLib;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Vector;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;

public class wekaDiscretizeMethod {

	private boolean isSupervise = true;
	private File datafile = null;
	private Instances originaldata = null;
	private Instances discretizedata = null;
	public wekaDiscretizeMethod(File datafile, boolean isSupervise) throws Exception, IOException{
		this.datafile = datafile;
		this.isSupervise = isSupervise;
		if(datafile!=null){
			this.originaldata = new Instances(new FileReader(this.datafile));
			this.originaldata.setClassIndex(originaldata.numAttributes()-1); //设置决策属性索引
			
		}
		if(this.isSupervise){
		    Discretize mydiscretize = new Discretize();
		    mydiscretize.setInputFormat(this.originaldata);
		    this.discretizedata = Filter.useFilter(this.originaldata, mydiscretize);
		}
		else{
			this.discretizedata =  null;
		}
	}

	public Instances getOriginalData(){
		return this.originaldata;
	}
	public Instances getDiscretizeData(){
		return this.discretizedata;
	}
	public static Instances getSelectedInstances(Instances OriginalData, boolean[] B) throws Exception{
		int[] reAttr = Utils.boolean2remove(B);
		
		Remove m_removeFilter = new Remove();
		m_removeFilter.setAttributeIndicesArray(reAttr);
		m_removeFilter.setInvertSelection(false);
    	m_removeFilter.setInputFormat(OriginalData);   
    	Instances newData = Filter.useFilter(OriginalData, m_removeFilter);
    	newData.setClassIndex( newData.numAttributes() - 1 ); //重新设置决策属性索引
    	
    	return newData;
	}
	
	public static Instances getUnSelectedInstances(Instances OriginalData, boolean[] B) throws Exception{
		if(Utils.isAllFalse(B))
			return OriginalData;
		int[] reAttr = Utils.boolean2select(B);
		//System.out.println(Arrays.toString(reAttr));
		
		Remove m_removeFilter = new Remove();
		m_removeFilter.setAttributeIndicesArray(reAttr);
		m_removeFilter.setInvertSelection(false);
    	m_removeFilter.setInputFormat(OriginalData);   
    	Instances newData = Filter.useFilter(OriginalData, m_removeFilter);
    	newData.setClassIndex( newData.numAttributes() - 1 ); //重新设置决策属性索引
    	
    	return newData;
	}
	/**
	 * @param args
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, IOException {
		// TODO Auto-generated method stub
		String df = "E://new//Xreducer//data//colic.arff";
		Instances data = new Instances(new FileReader(new File(df)));
		data.setClassIndex(data.numAttributes()-1);
		System.out.println(data.numInstances());
		boolean[] B = Utils.Instances2FullBoolean(data);
		B[0]=false;

		filterDataSet(data,B);
		for(int i=0;i<data.numInstances();++i)
			System.out.println(data.instance(i).toString());
		System.out.println(data.numInstances());
	}

	public static Instances filterDataSet(Instances data, boolean[] B) {
		// TODO Auto-generated method stub
		int N = data.numInstances();
		boolean[] flag = new boolean[N];
		for(int i=0;i<N;++i){
			if(!flag[i]){
				for(int j=i+1;j<N;++j){
					if(!flag[j]&&instancesIsEqual(data,B,i,j)){
						flag[j]=true;
					}
				}
			}
		}
		return removeFlagData(data,flag);
	}
	public static boolean instancesIsEqual(Instances data, boolean[] B, int i, int j) {
		// TODO Auto-generated method stub
		boolean isequal = true;
		int K = B.length-1;//不包括决策属性
		for(int k=0;k<K;++k){
			if(!B[k]&&data.instance(i).isMissing(k)&&data.instance(j).isMissing(k)) //miss值算作相同属性值
				continue;
			if(!B[k]&&data.instance(i).value(k)!=data.instance(j).value(k)){ //miss值算作不同属性值
				isequal = false;
				break;
			}
		}
		return isequal;
	}
	public static Instances removeFlagData(Instances data, boolean[] flag){
		int N = flag.length;
		Vector<Integer> equalIndex = new Vector<Integer>();
		for(int i=0;i<N;++i){
			if(flag[i])
				equalIndex.add(i);
		}
		int cnt = 0;
		for(int i=0;i<equalIndex.size();++i){
			int index = equalIndex.get(i);
			data.delete(index-cnt);
			cnt++;
		}
		return data;
		
		
	}
}
