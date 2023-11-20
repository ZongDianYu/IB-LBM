# Study of the Force Interaction between Slender Elastic Objects and Surrounding Fluid Using Immersed Boundary–Lattice Boltzmann Method
本研究以沉浸邊界-晶格玻茲曼法(IB-LBM)開發一套關於細長彈性體的流固耦合數值模型，藉由 C 語言與 NVIDIA 推出的CUDA，透過 GPU 來實現平行加速運算並探討細長彈性體於二維流場之流力作用。

## Motivation
流固耦合是描述流體力學與結構力學之間的多物理場相互作用，不管是大自然或是各種領域中都有廣泛的應用。以蝴蝶飛行為例，其翅膀具高度可撓性，在拍撲飛行時周圍流體所產生之作用力會對翅膀造成不能忽視的形變，而適度地形變有助於提高升力與降低阻力，同時也能增強其飛行穩定性。

## Method
現階段基於 body-conformal 網格的數值方法在處理複雜幾何邊界的形變與移動時，會有網格生成而增加計算成本、翅膀交疊使網格過密導致數值耗散等等挑戰。因此本研究在流體流動部分使用晶格波茲曼法，與傳統計算流體力學如有限元素法相比更簡單快速且具高度可平行化；彈性體使用沉浸邊界法，通過引入邊界變形力來模擬固體對流體的影響以克服複雜界面網格建置之困難。圖一為 IB-LBM 算法流程示意圖。
<div align = center>
<img src="https://github.com/ZongDianYu/IB-LBM/blob/main/figure/process.png">
</div>

## Result
圖二為雷諾數為 50 的圓柱繞流流場速度變化圖。模擬結果得到渦流從圓柱兩側脫離和卡門渦街發生的臨界雷諾數都與實驗數據接近。
<div align = center>
<img src="https://github.com/ZongDianYu/IB-LBM/blob/main/figure/Re50.gif" width="500" height="375">
</div>

圖三為多條彈性體與控制彈性體方向的流場速度變化圖。
<div align = center>
<img src="https://github.com/ZongDianYu/IB-LBM/blob/main/figure/move.gif" width="500" height="375">
</div>
