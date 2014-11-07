package cluster;

import java.awt.Color;
import java.awt.Font;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Vector;


import org.apache.poi.ss.usermodel.WorkbookFactory;
import org.apache.poi.xssf.usermodel.XSSFCell;
import org.apache.poi.xssf.usermodel.XSSFRow;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.HorizontalAlignment;
import org.jfree.ui.RectangleInsets;
import org.jfree.ui.VerticalAlignment;

import cluster.Myploter;

public class excelTopic {
	static final int methodnum=4;
	static final String[] methodname={"K-mode","SA-Kmode-W","SA-kmode-C","SA-kmode-WC"};
	static XSSFWorkbook book;
	static XSSFSheet sheet;
	static XSSFRow row;
	
	public static JFreeChart drawvaluevec(Vector<Vector<Double>> valueseq,String[] namestr,String dataname,String xlab,String ylab,double[] x)
			throws Exception {
		if (valueseq.size() != namestr.length)
			return null;
		XYSeries[] plotdata = new XYSeries[namestr.length];
		XYSeriesCollection plotdataset = new XYSeriesCollection();
		for (int i = 0; i < plotdata.length; i++) {
			plotdata[i] = new XYSeries(namestr[i]);
			for (int j = 0; j < valueseq.elementAt(i).size(); j++) {
				plotdata[i].add(x[j], valueseq.elementAt(i).elementAt(j).doubleValue());
			}
			plotdataset.addSeries(plotdata[i]);
		}
		JFreeChart chart = ChartFactory.createScatterPlot(dataname, // chart															// title
				xlab, // x axis label
				ylab, // y axis label
				plotdataset, // data
				PlotOrientation.VERTICAL, true, // include legend
				true, // tooltips
				false // urls
				);
		XYPlot plot = (XYPlot) chart.getPlot();
		XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();
		renderer.setSeriesLinesVisible(0, true);
		
//		renderer.setSeriesPaint(2, Color.DARK_GRAY);
//		renderer.setSeriesPaint(3, Color.GREEN);
//		renderer.setSeriesPaint(4, Color.ORANGE);
//		renderer.setSeriesPaint(5, Color.MAGENTA);
		
//		renderer.setSeriesPaint(0, Color.BLUE);
//		renderer.setSeriesPaint(1, Color.red);
		plot.setRenderer(renderer);
		plot.getDomainAxis().setLabelFont(new Font("黑体",0,18));
		plot.getDomainAxis().setTickLabelFont(new Font("黑体",0,18));
		plot.getRangeAxis().setLabelFont(new Font("黑体",0,18));
		plot.getRangeAxis().setTickLabelFont(new Font("黑体",0,18));
		plot.setBackgroundPaint(Color.WHITE);
		
		//plot.get
		chart.setBackgroundPaint(Color.WHITE);
		chart.getTitle().setFont(new Font("黑体",0,25));
		chart.getLegend(0).setItemFont(new Font("黑体",0,22));
		chart.getLegend(0).setMargin(new RectangleInsets(0,120,20,120));
		chart.getLegend(0).setItemLabelPadding(new RectangleInsets(1,1,1,1));
		chart.getLegend(0).setPadding(new RectangleInsets(5,5,5,5));
		return chart;
	}
	
	public static Vector<Vector<Double>> readdata(XSSFSheet sht){
		Vector<Vector<Double>> valueseq=new Vector<Vector<Double>>();
		for(int i=0;i<methodnum;i++){
			Vector<Double> v=new Vector<Double>();
			valueseq.add(v);
		}
		for(int i=1;i<101;i++){
			row=sht.getRow(i);
			for(int j=0;j<methodnum;j++){
				XSSFCell cell=row.getCell(j);
				if(cell!=null){
					valueseq.elementAt(j).add(cell.getNumericCellValue());
				}
			}
		}
		
		return valueseq;
	}

	
	
	public static void main(String[] args) throws Exception {
			
		double[] x = new double[100];
		for(int i=0;i<x.length;i++)
			x[i]=i;

		String[] datastr = {"data15_3_3_10_10_500","data50_5_5_10_5_300","zoo","vote","car",
				"lymph","soybean", "mushroom"};

		for(int i=0;i<datastr.length;i++){
		String dataname=datastr[i];
		InputStream input = new FileInputStream("ans\\readexcelkmode\\Kmode_"+dataname+".xlsx");
		
		book=new XSSFWorkbook(input);
		
		Vector<Vector<Double>> valueseq=null;
		
		sheet=book.getSheetAt(0);
		valueseq=readdata(sheet);
		JFreeChart chart=drawvaluevec(valueseq,methodname,dataname,"","J(W,Z)",x);
		Myploter.saveAsFile(chart, "ans\\" + dataname + "_cmp_bf.jpg", 800, 600);		
		}
		
	}

}
