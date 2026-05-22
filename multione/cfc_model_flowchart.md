
:::mermaid  
flowchart LR
    X["Sequence inputs (xt)"] --> C["Concat"]
    T["Timeless features"] --> C
    TS["timespans"] -->|"dt_fwd / -dt_bwd"| RNN

    subgraph MixedMemory
      C --> LSTM["LSTMCell"]
      C --> CfC["CfCCell (continuous-time)"]
      LSTM --> RNN["Shared hidden state h"]
      CfC --> RNN
    end

    RNN -->|"forward"| Hfwd["h_fwd @ last t"]
    RNN -->|"backward"| Hbwd["h_bwd @ first t"]

    Hfwd --> Ffwd["fc_fwd MLP"] --> Yfwd["forward prediction"]
    Hbwd --> Fbwd["fc_bwd MLP"] --> Ybwd["backward prediction"]

    Yfwd --> OUT["Stacked output"]
    Ybwd --> OUT
:::