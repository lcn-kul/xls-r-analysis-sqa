# xls-r-analysis-sqa
Analysis of XLS-R for Speech Quality Assessment

### 1.2. Example Audio Segments

<details>
  <summary>🔊
  
  **Excellent** (MOS = 4.808)
  </summary>
  
  <table>
      <thead>
          <tr>
              <th>Audio Sample</th>
              <th>Model</th>
              <th>Prediction</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan=3><video src="https://user-images.githubusercontent.com/32679237/235354126-444c44ce-3e39-46da-8b4e-647e64ee243a.mp4"> |</td>
              <td align=center>DNSMOS</td>
              <td align=center>3.699</td>
          </tr>
          <tr>
              <td align=center>MFCC Transformer</td>
              <td align=center>3.231</td>
          </tr>
          <tr>
              <td align=center>XLS-R 1B Layer41 <br /> Transformer</td>
              <td align=center>4.126</td>
          </tr>
      </tbody>
  </table>
  
</details>

<details>
  <summary>🔊
  
  **Good** (MOS = 4.104)
  </summary>

  <table>
      <thead>
          <tr>
              <th>Audio Sample</th>
              <th>Model</th>
              <th>Prediction</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan=3><video src="https://user-images.githubusercontent.com/32679237/235354278-277152e2-da3e-48aa-b21c-1ddee3e9f0cc.mp4"> |</td>
              <td align=center>DNSMOS</td>
              <td align=center>3.269</td>
          </tr>
          <tr>
              <td align=center>MFCC Transformer</td>
              <td align=center>3.276</td>
          </tr>
          <tr>
              <td align=center>XLS-R 1B Layer41 <br /> Transformer</td>
              <td align=center>3.260</td>
          </tr>
      </tbody>
  </table>

  
  
</details>

<details>
  <summary>🔊
  
  **Fair** (MOS = 3.168)
  </summary>

  <table>
      <thead>
          <tr>
              <th>Audio Sample</th>
              <th>Model</th>
              <th>Prediction</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan=3><video src="https://user-images.githubusercontent.com/32679237/235358366-df15fb96-7926-4a8e-8d06-cc1833aec3e3.mp4"> |</td>
              <td align=center>DNSMOS</td>
              <td align=center>3.309</td>
          </tr>
          <tr>
              <td align=center>MFCC Transformer</td>
              <td align=center>3.515</td>
          </tr>
          <tr>
              <td align=center>XLS-R 1B Layer41 <br /> Transformer</td>
              <td align=center>3.404</td>
          </tr>
      </tbody>
  </table>


  

</details>

<details>
  <summary>🔊
  
  **Poor** (MOS = 2.240)
  </summary>

  <table>
      <thead>
          <tr>
              <th>Audio Sample</th>
              <th>Model</th>
              <th>Prediction</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan=3><video src="https://user-images.githubusercontent.com/32679237/235354283-7d765c2f-0e78-48aa-8ac2-26640b09eaf4.mp4"> |</td>
              <td align=center>DNSMOS</td>
              <td align=center>2.553</td>
          </tr>
          <tr>
              <td align=center>MFCC Transformer</td>
              <td align=center>1.794</td>
          </tr>
          <tr>
              <td align=center>XLS-R 1B Layer41 <br /> Transformer</td>
              <td align=center>2.031</td>
          </tr>
      </tbody>
  </table>
  
</details>

<details>
  <summary>🔊
  
  **Bad** (MOS = 1.416)
  </summary>
  
  <table>
      <thead>
          <tr>
              <th>Audio Sample</th>
              <th>Model</th>
              <th>Prediction</th>
          </tr>
      </thead>
      <tbody>
          <tr>
              <td rowspan=3><video src="https://user-images.githubusercontent.com/32679237/235355743-2ebdb1bf-e9aa-4538-a3fe-acd9633e6443.mp4"> |</td>
              <td align=center>DNSMOS</td>
              <td align=center>2.553</td>
          </tr>
          <tr>
              <td align=center>MFCC Transformer</td>
              <td align=center>1.794</td>
          </tr>
          <tr>
              <td align=center>XLS-R 1B Layer41 <br /> Transformer</td>
              <td align=center>2.031</td>
          </tr>
      </tbody>
  </table>
  
</details>

