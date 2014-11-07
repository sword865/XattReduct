package UFS;

import java.io.FileReader;
import java.text.SimpleDateFormat;
import java.util.Arrays;

import helpLib.Utils;
import Xreducer_fuzzy.ITStyle_Lukasiewicz;
import Xreducer_fuzzy.ImplicatorTnormStyle;
import Xreducer_fuzzy.MStyle_FCBF;
import Xreducer_fuzzy.SStyle_MaxMin;
import Xreducer_fuzzy.SimilarityStyle;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public abstract class FSmethod {

	
	public Instances m_data = null;
	public String m_process = "";
	public double m_useTime = 0.0;
	public int m_numRed = 0;
	public int[] m_selectAtt = null;
	public String algname = "";
	
	public FSmethod (Instances data){
		this.m_data = data;
	}

	public abstract boolean[] getReduceAtt();
	public int[] getSelectedAtt(){
		boolean[] red = getReduceAtt();
		red[red.length-1]=true;  //选择决策属性
		SimpleDateFormat tempDate = new SimpleDateFormat("HH:mm:ss");
		String datetime = tempDate.format(new java.util.Date());
		 
		System.out.println(this.algname+"Success!+At "+datetime);
		return Utils.boolean2select_wDec(red);
	}
	public  String getInformation(){
		String str = this.algname+"->所用时间:" + Utils.doubleFormat("0.0000", this.m_useTime)+"s  约简个数："+this.m_numRed+"\n"+this.m_process;
		str += "最终约简:"+Arrays.toString(this.m_selectAtt);
		System.out.println(str);
		return str;
	}
 
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {

	}

}
